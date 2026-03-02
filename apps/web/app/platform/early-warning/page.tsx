import { BackButton } from "@/components/back-button";

export default function EarlyWarningSystem() {
    return (
        <div className="min-h-screen">
            <div className="mx-auto max-w-4xl px-6 py-24">
                <BackButton />

                <h1 className="mt-8 text-4xl font-bold md:text-5xl">
                    Early Warning System
                </h1>
                <p className="mt-4 text-xl text-muted-foreground">
                    Detect instability signals before they become crises
                </p>

                <div className="mt-12 space-y-8">
                    <section>
                        <h2 className="text-2xl font-semibold">The Cost of Late Detection</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Every governance crisis has a prehistory—a period when warning signs were present but not recognized, when intervention would have been straightforward but wasn't attempted. The difference between managing an issue and fighting a crisis often comes down to weeks or even days of early detection.
                        </p>
                        <p className="mt-4 text-lg leading-relaxed">
                            The Early Warning System exists to close this gap. It continuously monitors for the subtle patterns that precede instability, alerting leadership while there's still time for measured, proportionate response.
                        </p>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">How Instability Announces Itself</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Instability rarely appears suddenly. It builds through recognizable patterns: accelerating trends, breaking correlations, emerging clusters, and threshold crossings. Our system watches for these signatures across multiple domains simultaneously.
                        </p>

                        <div className="mt-6 space-y-6">
                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Anomaly Detection</h3>
                                <p className="mt-3 leading-relaxed">
                                    Statistical deviations from established baselines, accounting for seasonal patterns, day-of-week effects, and known events. The system learns what's normal for your jurisdiction and alerts when conditions diverge meaningfully.
                                </p>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Trend Acceleration</h3>
                                <p className="mt-3 leading-relaxed">
                                    Gradual changes become concerning when they accelerate. The system monitors rates of change across indicators, flagging when deterioration is speeding up—the moment when intervention becomes urgent.
                                </p>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Correlation Breakdown</h3>
                                <p className="mt-3 leading-relaxed">
                                    When historically related indicators decouple, it often signals systemic stress. Economic growth without employment gains, rising incomes with falling consumer confidence—these disconnections reveal underlying instability.
                                </p>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Geographic Clustering</h3>
                                <p className="mt-3 leading-relaxed">
                                    Problems that appear scattered become concerning when they cluster geographically. The system identifies spatial patterns that suggest localized deterioration before it spreads.
                                </p>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Cross-Domain Convergence</h3>
                                <p className="mt-3 leading-relaxed">
                                    The most serious instabilities show up across multiple domains simultaneously. When economic stress coincides with social tension and public safety concerns in the same area, the system recognizes the convergence and escalates the alert.
                                </p>
                            </div>
                        </div>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">Intelligent Alerting</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            The system is designed to inform, not overwhelm. Alerts are prioritized by severity, confidence, and time-sensitivity. You receive actionable intelligence, not raw data dumps.
                        </p>
                        <ul className="mt-4 space-y-3 text-lg">
                            <li className="flex items-start">
                                <span className="mr-3 mt-1 text-primary">•</span>
                                <span><strong>Severity scoring:</strong> Understand how concerning each signal is relative to historical patterns</span>
                            </li>
                            <li className="flex items-start">
                                <span className="mr-3 mt-1 text-primary">•</span>
                                <span><strong>Confidence levels:</strong> Know how certain the system is about each detection</span>
                            </li>
                            <li className="flex items-start">
                                <span className="mr-3 mt-1 text-primary">•</span>
                                <span><strong>Time windows:</strong> Understand how quickly the situation may evolve</span>
                            </li>
                            <li className="flex items-start">
                                <span className="mr-3 mt-1 text-primary">•</span>
                                <span><strong>Contextual briefings:</strong> Each alert includes relevant background and historical context</span>
                            </li>
                        </ul>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">From Detection to Action</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Early warning is valuable only if it enables early action. The system integrates directly with response coordination, allowing you to move from alert to intervention seamlessly. Every warning includes recommended response options based on similar historical situations.
                        </p>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">Proven Performance</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Jurisdictions using the Early Warning System detect emerging issues an average of 18 days earlier than traditional monitoring methods. This lead time is the difference between prevention and crisis management.
                        </p>
                        <p className="mt-4 text-lg leading-relaxed">
                            More importantly, early detection enables proportionate response. Small interventions work when you catch problems early. The system gives you that opportunity.
                        </p>
                    </section>

                    <section className="rounded-lg bg-accent/10 p-8">
                        <h2 className="text-2xl font-semibold">See Problems Before They See You</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Learn how the Early Warning System can enhance your capacity for preventive governance. Schedule a consultation to discuss your jurisdiction's specific risk profile.
                        </p>
                        <button className="mt-6 rounded-lg bg-primary px-6 py-3 font-semibold text-primary-foreground hover:bg-primary/90">
                            Schedule Consultation
                        </button>
                    </section>
                </div>
            </div>
        </div>
    );
}
