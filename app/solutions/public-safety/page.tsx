import { BackButton } from "@/components/back-button";

export default function PublicSafety() {
    return (
        <div className="min-h-screen">
            <div className="mx-auto max-w-4xl px-6 py-24">
                <BackButton />

                <h1 className="mt-8 text-4xl font-bold md:text-5xl">
                    Public Safety Intelligence
                </h1>
                <p className="mt-4 text-xl text-muted-foreground">
                    From reactive policing to proactive safety governance
                </p>

                <div className="mt-12 space-y-8">
                    <section>
                        <h2 className="text-2xl font-semibold">Rethinking Public Safety</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Traditional public safety focuses on response: crimes occur, police respond, justice processes follow. This reactive model is necessary but insufficient. By the time response mechanisms activate, harm has already occurred.
                        </p>
                        <p className="mt-4 text-lg leading-relaxed">
                            Effective public safety governance requires understanding the conditions that generate insecurity before they manifest as incidents. This is the shift from policing events to governing safety conditions.
                        </p>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">What We Monitor</h2>
                        <div className="mt-6 space-y-6">
                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Crime Pattern Analysis</h3>
                                <p className="mt-3 leading-relaxed">
                                    Not just where crimes occur, but when, why, and how patterns are evolving. Identify emerging hotspots before they become entrenched. Detect shifts in crime types that signal changing conditions.
                                </p>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Social Stress Indicators</h3>
                                <p className="mt-3 leading-relaxed">
                                    Economic hardship, social fragmentation, and community tension often precede safety deterioration. Monitor these upstream indicators to identify areas at risk before crime rates rise.
                                </p>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Environmental Design</h3>
                                <p className="mt-3 leading-relaxed">
                                    Physical environment affects safety. Poor lighting, abandoned properties, inadequate public space—these create conditions for insecurity. Track environmental factors alongside incident data.
                                </p>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Community Confidence</h3>
                                <p className="mt-3 leading-relaxed">
                                    Public perception of safety matters as much as objective measures. When communities feel unsafe, behavior changes in ways that can actually reduce safety. Monitor confidence alongside incidents.
                                </p>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Service Accessibility</h3>
                                <p className="mt-3 leading-relaxed">
                                    Response times, service coverage, and accessibility patterns. Ensure that safety services reach all communities equitably. Identify gaps before they become grievances.
                                </p>
                            </div>
                        </div>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">Preventive Intervention</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            When the system identifies deteriorating safety conditions, it recommends proportionate preventive interventions:
                        </p>
                        <ul className="mt-4 space-y-3 text-lg">
                            <li className="flex items-start">
                                <span className="mr-3 mt-1 text-primary">•</span>
                                <span><strong>Environmental improvements:</strong> Lighting, maintenance, public space activation</span>
                            </li>
                            <li className="flex items-start">
                                <span className="mr-3 mt-1 text-primary">•</span>
                                <span><strong>Social programs:</strong> Youth engagement, economic support, community building</span>
                            </li>
                            <li className="flex items-start">
                                <span className="mr-3 mt-1 text-primary">•</span>
                                <span><strong>Targeted enforcement:</strong> Focused deterrence in emerging hotspots</span>
                            </li>
                            <li className="flex items-start">
                                <span className="mr-3 mt-1 text-primary">•</span>
                                <span><strong>Community engagement:</strong> Dialogue, partnership, co-production of safety</span>
                            </li>
                        </ul>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">Evidence-Based Resource Allocation</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Public safety resources are always constrained. The system helps allocate them where they'll have the greatest impact, based on risk assessment, historical effectiveness, and current conditions. This is how you do more with what you have.
                        </p>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">Equity and Legitimacy</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            The system monitors for disparities in both safety outcomes and service delivery. Are some communities systematically less safe? Are some receiving disproportionate enforcement attention? These patterns undermine both effectiveness and legitimacy.
                        </p>
                        <p className="mt-4 text-lg leading-relaxed">
                            Equitable safety is not just a moral imperative—it's a practical requirement for sustainable security.
                        </p>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">Measured Impact</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Jurisdictions using our Public Safety Intelligence report 25% reduction in crime rates, 40% improvement in community confidence, and more equitable distribution of both safety outcomes and service delivery. More importantly, they report shifting from crisis response to condition management.
                        </p>
                    </section>

                    <section className="rounded-lg bg-accent/10 p-8">
                        <h2 className="text-2xl font-semibold">Govern Safety Conditions, Not Just Crime Events</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Learn how Public Safety Intelligence can enhance your capacity for preventive safety governance. Schedule a consultation focused on your jurisdiction's safety challenges.
                        </p>
                        <button className="mt-6 rounded-lg bg-primary px-6 py-3 font-semibold text-primary-foreground hover:bg-primary/90">
                            Schedule Safety Consultation
                        </button>
                    </section>
                </div>
            </div>
        </div>
    );
}
