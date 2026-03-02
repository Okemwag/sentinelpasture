import { BackButton } from "@/components/back-button";

export default function Research() {
    return (
        <div className="min-h-screen">
            <div className="mx-auto max-w-4xl px-6 py-24">
                <BackButton />

                <h1 className="mt-8 text-4xl font-bold md:text-5xl">
                    Research Papers
                </h1>
                <p className="mt-4 text-xl text-muted-foreground">
                    Academic foundations of governance intelligence
                </p>

                <div className="mt-12 space-y-8">
                    <section>
                        <h2 className="text-2xl font-semibold">Research-Grounded Practice</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Our approach to governance intelligence is grounded in rigorous academic research across multiple disciplines: public administration, complexity science, causal inference, organizational theory, and political economy.
                        </p>
                        <p className="mt-4 text-lg leading-relaxed">
                            We publish our research openly and engage with the academic community to continuously refine our methods and validate our approaches.
                        </p>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">Research Areas</h2>
                        <div className="mt-6 space-y-6">
                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Early Warning Systems</h3>
                                <p className="mt-3 leading-relaxed">
                                    Methods for detecting instability signals, pattern recognition in governance data, and validation of early warning indicators.
                                </p>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Causal Inference in Governance</h3>
                                <p className="mt-3 leading-relaxed">
                                    Applying advanced causal inference methods to policy evaluation, intervention design, and evidence-based governance.
                                </p>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Complex Systems and Governance</h3>
                                <p className="mt-3 leading-relaxed">
                                    Understanding governance challenges as complex adaptive systems, including emergence, feedback loops, and non-linear dynamics.
                                </p>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Institutional Capacity</h3>
                                <p className="mt-3 leading-relaxed">
                                    Research on how institutions learn, adapt, and build capacity for proactive governance.
                                </p>
                            </div>
                        </div>
                    </section>

                    <section className="rounded-lg bg-accent/10 p-8">
                        <h2 className="text-2xl font-semibold">Access Research Publications</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Browse our research library and access published papers, working papers, and technical reports.
                        </p>
                        <button className="mt-6 rounded-lg bg-primary px-6 py-3 font-semibold text-primary-foreground hover:bg-primary/90">
                            Browse Research Library
                        </button>
                    </section>
                </div>
            </div>
        </div>
    );
}
