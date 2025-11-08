package sk.ainet.exec.tensor.ops

import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import jdk.incubator.vector.VectorSpecies

/**
 * Thin wrapper over the JDK Vector API. Isolate incubator usage here so call sites stay clean
 * and future API changes are easier to adapt.
 */
internal object JvmVectorKernels {
    private val floatSpecies: VectorSpecies<Float> = FloatVector.SPECIES_PREFERRED

    // region Elementwise
    fun binaryFloat(
        a: FloatArray,
        b: FloatArray,
        out: FloatArray,
        length: Int,
        op: (FloatVector, FloatVector) -> FloatVector,
        scalarOp: (Float, Float) -> Float,
    ) {
        var index = 0
        val step = floatSpecies.length()
        val loopBound = floatSpecies.loopBound(length)
        while (index < loopBound) {
            val va = FloatVector.fromArray(floatSpecies, a, index)
            val vb = FloatVector.fromArray(floatSpecies, b, index)
            op(va, vb).intoArray(out, index)
            index += step
        }
        while (index < length) {
            out[index] = scalarOp(a[index], b[index])
            index++
        }
    }

    fun unaryFloat(
        input: FloatArray,
        out: FloatArray,
        length: Int,
        op: (FloatVector) -> FloatVector,
        scalarOp: (Float) -> Float,
    ) {
        var index = 0
        val step = floatSpecies.length()
        val loopBound = floatSpecies.loopBound(length)
        while (index < loopBound) {
            val v = FloatVector.fromArray(floatSpecies, input, index)
            op(v).intoArray(out, index)
            index += step
        }
        while (index < length) {
            out[index] = scalarOp(input[index])
            index++
        }
    }
    // endregion

    // region Reductions
    fun reduceAllSumFloat(input: FloatArray, length: Int): Float {
        var index = 0
        val step = floatSpecies.length()
        val loopBound = floatSpecies.loopBound(length)
        var accVec = FloatVector.zero(floatSpecies)
        while (index < loopBound) {
            val v = FloatVector.fromArray(floatSpecies, input, index)
            accVec = accVec.add(v)
            index += step
        }
        var acc = accVec.reduceLanes(VectorOperators.ADD)
        while (index < length) {
            acc += input[index]
            index++
        }
        return acc
    }
    // endregion

    // region Matmul (naive + vectorized inner product)
    fun matmulFloat(
        aRows: Int,
        aCols: Int,
        bCols: Int,
        a: FloatArray,
        b: FloatArray,
        out: FloatArray,
    ) {
        // b is expected in normal layout; we transpose internally for cache-friendly access
        val transposedB = FloatArray(bCols * aCols)
        // original b has dimensions aCols x bCols
        for (row in 0 until aCols) {
            val srcOffset = row * bCols
            for (col in 0 until bCols) {
                transposedB[col * aCols + row] = b[srcOffset + col]
            }
        }

        val step = floatSpecies.length()
        val loopBound = floatSpecies.loopBound(aCols)
        for (row in 0 until aRows) {
            val aOffset = row * aCols
            for (col in 0 until bCols) {
                val bOffset = col * aCols
                var idx = 0
                var accVec = FloatVector.zero(floatSpecies)
                while (idx < loopBound) {
                    val va = FloatVector.fromArray(floatSpecies, a, aOffset + idx)
                    val vb = FloatVector.fromArray(floatSpecies, transposedB, bOffset + idx)
                    accVec = accVec.add(va.mul(vb))
                    idx += step
                }
                var acc = accVec.reduceLanes(VectorOperators.ADD)
                while (idx < aCols) {
                    acc += a[aOffset + idx] * transposedB[bOffset + idx]
                    idx++
                }
                out[row * bCols + col] = acc
            }
        }
    }

    /**
     * Blocked (tiled) matmul with vectorized inner products. Designed for small/medium matrices.
     * Tiles of 8x8 generally perform well across x86/ARM with preferred species.
     */
    fun matmulFloatBlocked(
        aRows: Int,
        aCols: Int,
        bCols: Int,
        a: FloatArray,
        b: FloatArray,
        out: FloatArray,
        tileM: Int = 8,
        tileN: Int = 8,
        tileK: Int = 128,
    ) {
        // Transpose B for contiguous access along K
        val bt = FloatArray(bCols * aCols)
        for (k in 0 until aCols) {
            val src = k * bCols
            for (n in 0 until bCols) {
                bt[n * aCols + k] = b[src + n]
            }
        }
        val step = floatSpecies.length()
        val mBlocks = (aRows + tileM - 1) / tileM
        val nBlocks = (bCols + tileN - 1) / tileN
        val kBlocks = (aCols + tileK - 1) / tileK
        for (bm in 0 until mBlocks) {
            val mStart = bm * tileM
            val mEnd = minOf(mStart + tileM, aRows)
            for (bn in 0 until nBlocks) {
                val nStart = bn * tileN
                val nEnd = minOf(nStart + tileN, bCols)
                // Initialize C tile
                for (m in mStart until mEnd) {
                    val rowOff = m * bCols
                    for (n in nStart until nEnd) {
                        if (kBlocks == 0) {
                            out[rowOff + n] = 0f
                        } else if (bm == 0 && bn == 0) {
                            // ensure zeroing only once in simple flow; otherwise accumulate below
                            out[rowOff + n] = 0f
                        }
                    }
                }
                for (bk in 0 until kBlocks) {
                    val kStart = bk * tileK
                    val kEnd = minOf(kStart + tileK, aCols)
                    // Compute C[m, n] += A[m, k] * Bt[n, k] over kStart..kEnd
                    val loopBound = floatSpecies.loopBound(kEnd - kStart)
                    val kLen = kEnd - kStart
                    for (m in mStart until mEnd) {
                        val aBase = m * aCols + kStart
                        val cBase = m * bCols
                        for (n in nStart until nEnd) {
                            val btBase = n * aCols + kStart
                            var idx = 0
                            var accVec = FloatVector.zero(floatSpecies)
                            while (idx < loopBound) {
                                val va = FloatVector.fromArray(floatSpecies, a, aBase + idx)
                                val vb = FloatVector.fromArray(floatSpecies, bt, btBase + idx)
                                accVec = accVec.add(va.mul(vb))
                                idx += step
                            }
                            var acc = accVec.reduceLanes(VectorOperators.ADD)
                            while (idx < kLen) {
                                acc += a[aBase + idx] * bt[btBase + idx]
                                idx++
                            }
                            out[cBase + n] += acc
                        }
                    }
                }
            }
        }
    }
    // endregion
}
