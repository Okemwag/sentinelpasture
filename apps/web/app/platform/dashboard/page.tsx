import { BackButton } from "@/components/back-button";

export default function IntelligenceDashboard() {
    return (
        <div className="min-h-screen">
            <div className="mx-auto max-w-4xl px-6 py-24">
                <BackButton />

                <h1 className="mt-8 text-4xl font-bold md:text-5xl">
                    Intelligence Dashboard
                </h1>
                <p className="mt-4 text-xl text-muted-foreground">
                    Real-time governance monitoring for proactive leadership
                </p>

                <div className="mt-12 space-y-8">
                    <section>
                        <h2 className="text-2xl font-semibold">The Challenge of Modern Governance</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Regional bodies and state institutions face an unprecedented challenge: governing in an era where conditions change faster than traditional reporting cycles can capture. By the time quarterly reports reach decision-makers, the window for preventive action has often closed. Crises that could have been managed become emergencies that must be contained.
                        </p>
                        <p className="mt-4 text-lg leading-relaxed">
                            The Intelligence Dashboard transforms this paradigm. It doesn't replace human judgment—it amplifies it by ensuring leaders see emerging patterns before they crystallize into crises.
                        </p>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">What You See, When It Matters</h2>
                        <div className="mt-6 space-y-6">
                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Unified Situational Awareness</h3>
                                <p className="mt-3 leading-relaxed">
                                    Integrate data streams from public safety, economic indicators, social sentiment, infrastructure health, and environmental sensors into a single coherent view. No more siloed departments operating with incomplete pictures.
                                </p>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Signal-to-Noise Intelligence</h3>
                                <p className="mt-3 leading-relaxed">
                                    Our algorithms distinguish between routine fluctuations and meaningful deviations. You see what requires attention, not everything that's happening. This is the difference between data and intelligence.
                                </p>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Temporal Context</h3>
                                <p className="mt-3 leading-relaxed">
                                    Every indicator is presented with historical baselines, seasonal patterns, and trend trajectories. Understand not just what's happening, but whether it's normal, concerning, or critical given the context.
                                </p>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Geographic Intelligence</h3>
                                <p className="mt-3 leading-relaxed">
                                    Visualize patterns across your jurisdiction. Identify geographic clusters, spillover effects, and regional disparities that aggregate statistics obscure. See where conditions are deteriorating before they spread.
                                </p>
                            </div>
                        </div>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">Built for Leadership, Not Analysts</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            The dashboard is designed for executives who need to understand situations quickly and make decisions confidently. Every visualization answers a leadership question: What's changing? Why does it matter? What are my options?
                        </p>
                        <p className="mt-4 text-lg leading-relaxed">
                            Drill down when you need detail. Stay at the strategic level when you need perspective. The interface adapts to how leaders actually work.
                        </p>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">Real-World Impact</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Regional governments using the Intelligence Dashboard report 40% faster response times to emerging issues and 60% reduction in crisis escalations. More importantly, they report increased confidence in their ability to govern proactively rather than reactively.
                        </p>
                        <p className="mt-4 text-lg leading-relaxed">
                            This is governance intelligence: the capacity to see clearly, decide wisely, and act decisively before conditions force your hand.
                        </p>
                    </section>

                    <section className="rounded-lg bg-accent/10 p-8">
                        <h2 className="text-2xl font-semibold">Ready to Transform Your Situational Awareness?</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Schedule a demonstration tailored to your jurisdiction's specific challenges. See how the Intelligence Dashboard can enhance your capacity to govern conditions, not just respond to events.
                        </p>
                        <button className="mt-6 rounded-lg bg-primary px-6 py-3 font-semibold text-primary-foreground hover:bg-primary/90">
                            Request a Demo
                        </button>
                    </section>
                </div>
            </div>
        </div>
    );
}
