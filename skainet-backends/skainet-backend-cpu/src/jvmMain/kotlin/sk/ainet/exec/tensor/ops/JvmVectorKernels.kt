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
    // endregion
}
