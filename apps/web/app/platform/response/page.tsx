import { BackButton } from "@/components/back-button";

export default function ResponseCoordination() {
    return (
        <div className="min-h-screen">
            <div className="mx-auto max-w-4xl px-6 py-24">
                <BackButton />

                <h1 className="mt-8 text-4xl font-bold md:text-5xl">
                    Response Coordination
                </h1>
                <p className="mt-4 text-xl text-muted-foreground">
                    Coordinate proportionate interventions across agencies and jurisdictions
                </p>

                <div className="mt-12 space-y-8">
                    <section>
                        <h2 className="text-2xl font-semibold">The Coordination Challenge</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Modern governance problems rarely respect organizational boundaries. A public safety issue has economic dimensions. An environmental concern affects public health. A social tension requires coordinated response from multiple agencies.
                        </p>
                        <p className="mt-4 text-lg leading-relaxed">
                            Yet our response structures remain siloed. Agencies operate independently, often learning about each other's interventions after the fact. Duplication wastes resources. Gaps leave problems unaddressed. Conflicting approaches undermine effectiveness.
                        </p>
                        <p className="mt-4 text-lg leading-relaxed">
                            Response Coordination solves this problem by providing a unified platform for multi-agency intervention planning and execution.
                        </p>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">Unified Situational Picture</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            All participating agencies see the same intelligence, updated in real-time. When an issue is detected, every relevant stakeholder is immediately aware. No more telephone chains, email threads, or meetings to establish basic situational awareness.
                        </p>
                        <div className="mt-6 rounded-lg border p-6">
                            <h3 className="text-xl font-semibold">Shared Intelligence Layer</h3>
                            <p className="mt-3 leading-relaxed">
                                Detection alerts, analysis findings, and ongoing monitoring data are visible to all authorized agencies simultaneously. Everyone works from the same information base.
                            </p>
                        </div>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">Collaborative Response Planning</h2>
                        <div className="mt-6 space-y-6">
                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Role Clarity</h3>
                                <p className="mt-3 leading-relaxed">
                                    The system suggests which agencies should lead and support based on the nature of the issue, jurisdictional mandates, and available capabilities. Clear role assignment prevents both gaps and duplication.
                                </p>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Resource Visibility</h3>
                                <p className="mt-3 leading-relaxed">
                                    See what resources each agency can contribute, what's already committed elsewhere, and what gaps exist. Plan interventions based on actual capacity, not assumptions.
                                </p>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Timeline Coordination</h3>
                                <p className="mt-3 leading-relaxed">
                                    Sequence interventions appropriately. Some actions must precede others. Some should happen simultaneously. The system helps coordinate timing across multiple agencies.
                                </p>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Dependency Management</h3>
                                <p className="mt-3 leading-relaxed">
                                    Track dependencies between different agencies' actions. If one component is delayed, all dependent actions are automatically flagged for review.
                                </p>
                            </div>
                        </div>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">Proportionate Response</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            The system recommends intervention intensity based on issue severity, confidence in detection, and historical effectiveness of different response levels. Start with the minimum necessary intervention. Escalate if needed. This approach maximizes effectiveness while minimizing disruption and cost.
                        </p>
                        <ul className="mt-4 space-y-3 text-lg">
                            <li className="flex items-start">
                                <span className="mr-3 mt-1 text-primary">•</span>
                                <span><strong>Graduated response options:</strong> From monitoring to full intervention</span>
                            </li>
                            <li className="flex items-start">
                                <span className="mr-3 mt-1 text-primary">•</span>
                                <span><strong>Escalation triggers:</strong> Clear criteria for increasing response intensity</span>
                            </li>
                            <li className="flex items-start">
                                <span className="mr-3 mt-1 text-primary">•</span>
                                <span><strong>De-escalation protocols:</strong> How to step down when conditions improve</span>
                            </li>
                        </ul>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">Execution Tracking</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Once response is initiated, the system tracks execution in real-time. Are actions being taken as planned? Are they having the expected effect? Are conditions improving, stable, or deteriorating?
                        </p>
                        <p className="mt-4 text-lg leading-relaxed">
                            This continuous feedback enables adaptive response. If initial interventions aren't working, you know quickly and can adjust. If they're working better than expected, you can de-escalate sooner.
                        </p>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">Cross-Jurisdictional Coordination</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Many issues span multiple jurisdictions. The system facilitates coordination between neighboring regions, enabling joint response to shared challenges. Secure information sharing, coordinated planning, and unified execution—even when different governments are involved.
                        </p>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">Institutional Learning</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Every response is documented and evaluated. What worked? What didn't? Why? This institutional memory ensures that lessons learned in one situation inform responses to similar situations in the future.
                        </p>
                        <p className="mt-4 text-lg leading-relaxed">
                            Over time, your response capacity improves systematically. This is how organizations get better at governance.
                        </p>
                    </section>

                    <section className="rounded-lg bg-accent/10 p-8">
                        <h2 className="text-2xl font-semibold">Transform Coordination from Challenge to Capability</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Learn how Response Coordination can enhance your multi-agency intervention capacity. Schedule a demonstration focused on your coordination challenges.
                        </p>
                        <button className="mt-6 rounded-lg bg-primary px-6 py-3 font-semibold text-primary-foreground hover:bg-primary/90">
                            Schedule Demonstration
                        </button>
                    </section>
                </div>
            </div>
        </div>
    );
}
