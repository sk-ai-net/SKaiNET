package sk.ainet.lang.kan

/**
 * Legacy scratch DSL sample retained for reference only.
 * The body is commented out to keep the module buildable under explicit API.
 */
/*
definition<FP32, Float> {
    network(ctx) {
        input(1, "input")
        dense(16, "hidden-1") {
            weights { zeros() }
            bias { zeros() }
            activation = { tensor -> with(tensor) { relu() } }
        }
        dense(16, "hidden-2") {
            weights { zeros() }
            bias { zeros() }
            activation = { tensor -> with(tensor) { relu() } }
        }
        dense(1, "output") {
            weights { zeros() }
            bias { zeros() }
        }
    }
}
*/
