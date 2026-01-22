import { BackButton } from "@/components/back-button";

export default function SocialCohesion() {
    return (
        <div className="min-h-screen">
            <div className="mx-auto max-w-4xl px-6 py-24">
                <BackButton />

                <h1 className="mt-8 text-4xl font-bold md:text-5xl">
                    Social Cohesion Intelligence
                </h1>
                <p className="mt-4 text-xl text-muted-foreground">
                    Monitor and strengthen the social fabric of your jurisdiction
                </p>

                <div className="mt-12 space-y-8">
                    <section>
                        <h2 className="text-2xl font-semibold">The Foundation of Governability</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Social cohesion—the degree to which people trust each other, trust institutions, and feel connected to their communities—is the foundation of effective governance. When cohesion weakens, everything becomes harder: economic development, public safety, service delivery, collective action.
                        </p>
                        <p className="mt-4 text-lg leading-relaxed">
                            Yet social cohesion is rarely monitored systematically. We notice its absence only when it manifests as conflict, polarization, or institutional breakdown. By then, restoration is difficult and expensive.
                        </p>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">What We Monitor</h2>
                        <div className="mt-6 space-y-4">
                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Social Trust Indicators</h3>
                                <p className="mt-3 leading-relaxed">
                                    Interpersonal trust, institutional confidence, and civic participation patterns. Track how connected people feel to their communities and how much they trust governing institutions.
                                </p>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Community Fragmentation</h3>
                                <p className="mt-3 leading-relaxed">
                                    Geographic segregation, social isolation, and group polarization. Identify areas where communities are fragmenting before fragmentation hardens into division.
                                </p>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Grievance Accumulation</h3>
                                <p className="mt-3 leading-relaxed">
                                    Unresolved complaints, perceived injustices, and distributional inequities. These accumulate slowly but can trigger rapid deterioration when they reach critical mass.
                                </p>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Collective Efficacy</h3>
                                <p className="mt-3 leading-relaxed">
                                    The capacity of communities to solve problems collectively. Strong communities can address many challenges themselves. Weak communities require more intensive external intervention.
                                </p>
                            </div>
                        </div>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">Preventive Social Governance</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            When the system detects weakening cohesion, it recommends evidence-based interventions:
                        </p>
                        <ul className="mt-4 space-y-3 text-lg">
                            <li className="flex items-start">
                                <span className="mr-3 mt-1 text-primary">•</span>
                                <span><strong>Community dialogue initiatives:</strong> Structured engagement across dividing lines</span>
                            </li>
                            <li className="flex items-start">
                                <span className="mr-3 mt-1 text-primary">•</span>
                                <span><strong>Equity interventions:</strong> Address disparities that fuel resentment</span>
                            </li>
                            <li className="flex items-start">
                                <span className="mr-3 mt-1 text-primary">•</span>
                                <span><strong>Institutional responsiveness:</strong> Improve government accessibility and accountability</span>
                            </li>
                            <li className="flex items-start">
                                <span className="mr-3 mt-1 text-primary">•</span>
                                <span><strong>Community capacity building:</strong> Strengthen local problem-solving ability</span>
                            </li>
                        </ul>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">Long-Term Social Investment</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Social cohesion isn't built through crisis response—it requires sustained investment in the conditions that enable trust, connection, and collective action. The system helps you allocate social investment strategically, based on need and opportunity.
                        </p>
                    </section>

                    <section className="rounded-lg bg-accent/10 p-8">
                        <h2 className="text-2xl font-semibold">Strengthen the Social Foundation of Governance</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Learn how Social Cohesion Intelligence can help you monitor and strengthen community bonds. Schedule a consultation on social governance.
                        </p>
                        <button className="mt-6 rounded-lg bg-primary px-6 py-3 font-semibold text-primary-foreground hover:bg-primary/90">
                            Schedule Social Governance Consultation
                        </button>
                    </section>
                </div>
            </div>
        </div>
    );
}
