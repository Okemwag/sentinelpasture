import { BackButton } from "@/components/back-button";

export default function Environmental() {
    return (
        <div className="min-h-screen">
            <div className="mx-auto max-w-4xl px-6 py-24">
                <BackButton />

                <h1 className="mt-8 text-4xl font-bold md:text-5xl">
                    Environmental Risk Intelligence
                </h1>
                <p className="mt-4 text-xl text-muted-foreground">
                    Monitor environmental conditions that affect governance capacity
                </p>

                <div className="mt-12 space-y-8">
                    <section>
                        <h2 className="text-2xl font-semibold">Environmental Conditions as Governance Challenge</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Environmental risks—floods, droughts, heat waves, air quality crises, ecosystem degradation—directly affect your capacity to govern. They disrupt economic activity, threaten public health, damage infrastructure, and displace populations.
                        </p>
                        <p className="mt-4 text-lg leading-relaxed">
                            Climate change is accelerating these risks. Jurisdictions that governed environmental conditions effectively in the past face new challenges. Early detection and proactive management are no longer optional—they're essential for maintaining governability.
                        </p>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">Comprehensive Environmental Monitoring</h2>
                        <div className="mt-6 space-y-4">
                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Climate and Weather Patterns</h3>
                                <p className="mt-3 leading-relaxed">
                                    Temperature trends, precipitation patterns, extreme weather frequency. Understand how climate conditions in your jurisdiction are changing and what that means for governance.
                                </p>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Air and Water Quality</h3>
                                <p className="mt-3 leading-relaxed">
                                    Continuous monitoring of environmental quality indicators. Detect deterioration early enough to intervene before public health impacts occur.
                                </p>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Ecosystem Health</h3>
                                <p className="mt-3 leading-relaxed">
                                    Natural systems provide essential services—water filtration, flood control, climate regulation. Monitor ecosystem condition to ensure these services remain available.
                                </p>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Disaster Risk</h3>
                                <p className="mt-3 leading-relaxed">
                                    Flood risk, wildfire danger, drought vulnerability. Understand your exposure to environmental hazards and how that exposure is evolving.
                                </p>
                            </div>
                        </div>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">Early Warning and Preparedness</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            The system provides early warning of environmental risks, enabling proactive preparation rather than reactive crisis response. This approach saves lives, protects assets, and maintains continuity of governance.
                        </p>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">Climate Adaptation Planning</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Beyond managing current risks, the system supports long-term adaptation planning. As climate conditions change, governance strategies must adapt. Evidence-based adaptation is more effective and more affordable than reactive adjustment.
                        </p>
                    </section>

                    <section className="rounded-lg bg-accent/10 p-8">
                        <h2 className="text-2xl font-semibold">Strengthen Environmental Governance Capacity</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Learn how Environmental Risk Intelligence can enhance your capacity to manage environmental conditions proactively. Schedule an environmental governance consultation.
                        </p>
                        <button className="mt-6 rounded-lg bg-primary px-6 py-3 font-semibold text-primary-foreground hover:bg-primary/90">
                            Schedule Environmental Consultation
                        </button>
                    </section>
                </div>
            </div>
        </div>
    );
}
