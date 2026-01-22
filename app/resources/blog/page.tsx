import { BackButton } from "@/components/back-button";

export default function Blog() {
    return (
        <div className="min-h-screen">
            <div className="mx-auto max-w-4xl px-6 py-24">
                <BackButton />

                <h1 className="mt-8 text-4xl font-bold md:text-5xl">
                    Insights Blog
                </h1>
                <p className="mt-4 text-xl text-muted-foreground">
                    Perspectives on governance, leadership, and institutional capacity
                </p>

                <div className="mt-12 space-y-8">
                    <section>
                        <h2 className="text-2xl font-semibold">Governance Intelligence Insights</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Our blog explores the theory and practice of proactive governance, evidence-based policy, and institutional capacity building. Written by practitioners and researchers working at the intersection of governance and technology.
                        </p>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">Recent Posts</h2>
                        <div className="mt-6 space-y-6">
                            <article className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">From Reactive to Proactive: The Governance Transition</h3>
                                <p className="mt-2 text-sm text-muted-foreground">January 15, 2026 | 8 min read</p>
                                <p className="mt-3 leading-relaxed">
                                    Why traditional reactive governance is becoming untenable and what proactive governance requires in terms of capability, culture, and technology.
                                </p>
                                <button className="mt-4 text-primary hover:underline">Read More →</button>
                            </article>

                            <article className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">The Cost of Late Detection</h3>
                                <p className="mt-2 text-sm text-muted-foreground">January 8, 2026 | 6 min read</p>
                                <p className="mt-3 leading-relaxed">
                                    Quantifying the economic and social costs of detecting governance challenges late, and the value proposition of early warning systems.
                                </p>
                                <button className="mt-4 text-primary hover:underline">Read More →</button>
                            </article>

                            <article className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Evidence-Based Policy: Beyond the Rhetoric</h3>
                                <p className="mt-2 text-sm text-muted-foreground">December 28, 2025 | 10 min read</p>
                                <p className="mt-3 leading-relaxed">
                                    What evidence-based policy actually requires: causal understanding, rigorous evaluation, and institutional learning capacity.
                                </p>
                                <button className="mt-4 text-primary hover:underline">Read More →</button>
                            </article>
                        </div>
                    </section>

                    <section className="rounded-lg bg-accent/10 p-8">
                        <h2 className="text-2xl font-semibold">Subscribe to Insights</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Receive new posts and governance intelligence insights directly in your inbox.
                        </p>
                        <button className="mt-6 rounded-lg bg-primary px-6 py-3 font-semibold text-primary-foreground hover:bg-primary/90">
                            Subscribe to Blog
                        </button>
                    </section>
                </div>
            </div>
        </div>
    );
}
