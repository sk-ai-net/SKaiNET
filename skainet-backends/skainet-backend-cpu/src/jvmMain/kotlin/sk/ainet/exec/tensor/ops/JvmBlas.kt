package sk.ainet.exec.tensor.ops

import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.Linker
import java.lang.foreign.MemorySegment
import java.lang.foreign.SymbolLookup
import java.lang.invoke.MethodHandle
import java.lang.invoke.MethodType

/**
 * Minimal Panama FFM wrapper for BLAS `cblas_sgemm`.
 * This is optional and guarded by availability checks and config flags.
 */
internal object JvmBlas {
    private const val CBLAS_LAYOUT_ROW_MAJOR = 101
    private const val CBLAS_LAYOUT_COL_MAJOR = 102
    private const val CBLAS_NO_TRANS = 111
    private const val CBLAS_TRANS = 112

    private val linker: Linker = Linker.nativeLinker()

    @Volatile
    private var lookedUp: Boolean = false

    @Volatile
    private var sgemmHandle: MethodHandle? = null

    /** Try to find BLAS library and resolve cblas_sgemm symbol. */
    fun isAvailable(): Boolean {
        if (lookedUp) return sgemmHandle != null
        synchronized(this) {
            if (lookedUp) return sgemmHandle != null
            val libsToTry = listOf(
                // macOS Accelerate provides cblas_sgemm
                "Accelerate",
                // Common Linux names for OpenBLAS/ATLAS
                "blas",
                "openblas",
                "cblas"
            )
            val arena = Arena.ofAuto()
            var handle: MethodHandle? = null
            for (lib in libsToTry) {
                val lookup = runCatching { SymbolLookup.libraryLookup(lib, arena) }.getOrNull() ?: continue
                val symbol = lookup.find("cblas_sgemm").orElse(null) ?: continue
                handle = downcallSgemm(symbol)
                if (handle != null) break
            }
            sgemmHandle = handle
            lookedUp = true
            return sgemmHandle != null
        }
    }

    private fun downcallSgemm(symbol: MemorySegment): MethodHandle? {
        // cblas_sgemm layout:
        // void cblas_sgemm(const enum CBLAS_LAYOUT Layout, const enum CBLAS_TRANSPOSE TransA,
        //   const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
        //   const float alpha, const float *A, const int lda, const float *B, const int ldb,
        //   const float beta, float *C, const int ldc);
        val desc = FunctionDescriptor.ofVoid(
            java.lang.foreign.ValueLayout.JAVA_INT, // Layout
            java.lang.foreign.ValueLayout.JAVA_INT, // TransA
            java.lang.foreign.ValueLayout.JAVA_INT, // TransB
            java.lang.foreign.ValueLayout.JAVA_INT, // M
            java.lang.foreign.ValueLayout.JAVA_INT, // N
            java.lang.foreign.ValueLayout.JAVA_INT, // K
            java.lang.foreign.ValueLayout.JAVA_FLOAT, // alpha
            java.lang.foreign.ValueLayout.ADDRESS, // A
            java.lang.foreign.ValueLayout.JAVA_INT, // lda
            java.lang.foreign.ValueLayout.ADDRESS, // B
            java.lang.foreign.ValueLayout.JAVA_INT, // ldb
            java.lang.foreign.ValueLayout.JAVA_FLOAT, // beta
            java.lang.foreign.ValueLayout.ADDRESS, // C
            java.lang.foreign.ValueLayout.JAVA_INT, // ldc
        )
        return runCatching { linker.downcallHandle(symbol, desc) }.getOrNull()
    }

    /**
     * Row-major SGEMM (NoTrans x NoTrans) only. Other forms can be added as needed.
     * C := alpha * A x B + beta * C, where A is MxK, B is KxN, C is MxN.
     */
    fun sgemmRowMajorNN(m: Int, n: Int, k: Int, alpha: Float,
                        a: FloatArray, b: FloatArray, c: FloatArray,
                        lda: Int = k, ldb: Int = n, ldc: Int = n,
                        beta: Float = 0f): Boolean {
        val handle = sgemmHandle ?: return false
        val arena = Arena.ofConfined()
        val aSeg = MemorySegment.ofArray(a)
        val bSeg = MemorySegment.ofArray(b)
        val cSeg = MemorySegment.ofArray(c)
        return runCatching {
            handle.invoke(
                CBLAS_LAYOUT_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                m, n, k,
                alpha,
                aSeg, lda,
                bSeg, ldb,
                beta,
                cSeg, ldc,
            )
            // copy back into c in case the segment backed by a distinct array; MemorySegment.ofArray wraps the array
            true
        }.getOrElse { false }
    }
}
