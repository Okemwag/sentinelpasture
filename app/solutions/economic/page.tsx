import { BackButton } from "@/components/back-button";

export default function EconomicStability() {
    return (
        <div className="min-h-screen">
            <div className="mx-auto max-w-4xl px-6 py-24">
                <BackButton />

                <h1 className="mt-8 text-4xl font-bold md:text-5xl">
                    Economic Stability Intelligence
                </h1>
                <p className="mt-4 text-xl text-muted-foreground">
                    Monitor and manage economic conditions before they become crises
                </p>

                <div className="mt-12 space-y-8">
                    <section>
                        <h2 className="text-2xl font-semibold">Economic Governance in Volatile Times</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Regional economies face unprecedented volatility—global shocks, technological disruption, demographic shifts, climate impacts. Traditional economic indicators report what happened last quarter. By the time you see the data, conditions have changed.
                        </p>
                        <p className="mt-4 text-lg leading-relaxed">
                            Economic Stability Intelligence provides real-time monitoring of economic conditions, early detection of deterioration, and evidence-based intervention recommendations. This is how you govern economic stability rather than merely respond to economic crises.
                        </p>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">Comprehensive Economic Monitoring</h2>
                        <div className="mt-6 space-y-4">
                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Labor Market Dynamics</h3>
                                <p className="mt-3 leading-relaxed">
                                    Employment rates, job creation, wage trends, skill mismatches, and workforce participation—monitored continuously, not quarterly. Detect deterioration early enough to intervene effectively.
                                </p>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Business Health Indicators</h3>
                                <p className="mt-3 leading-relaxed">
                                    Business formation, failure rates, investment patterns, and sectoral shifts. Understand which parts of your economy are thriving and which are struggling.
                                </p>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Household Economic Security</h3>
                                <p className="mt-3 leading-relaxed">
                                    Income stability, debt levels, housing affordability, and cost-of-living pressures. Economic statistics look at aggregates; we monitor distribution and vulnerability.
                                </p>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Fiscal Sustainability</h3>
                                <p className="mt-3 leading-relaxed">
                                    Revenue trends, expenditure patterns, debt trajectories, and fiscal capacity. Ensure your economic interventions are sustainable.
                                </p>
                            </div>
                        </div>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">Early Warning Capabilities</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            The system detects economic deterioration through multiple signals: accelerating job losses, declining business confidence, rising household stress, weakening revenue growth. These patterns often appear months before they show up in official statistics.
                        </p>
                        <p className="mt-4 text-lg leading-relaxed">
                            Early detection enables early intervention—when problems are still manageable and solutions are still affordable.
                        </p>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">Evidence-Based Economic Policy</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            When economic stress is detected, the system recommends interventions based on causal analysis and historical effectiveness:
                        </p>
                        <ul className="mt-4 space-y-3 text-lg">
                            <li className="flex items-start">
                                <span className="mr-3 mt-1 text-primary">•</span>
                                <span><strong>Targeted support programs:</strong> Assistance where it's needed most</span>
                            </li>
                            <li className="flex items-start">
                                <span className="mr-3 mt-1 text-primary">•</span>
                                <span><strong>Business development initiatives:</strong> Sector-specific interventions</span>
                            </li>
                            <li className="flex items-start">
                                <span className="mr-3 mt-1 text-primary">•</span>
                                <span><strong>Workforce development:</strong> Skills training aligned with market needs</span>
                            </li>
                            <li className="flex items-start">
                                <span className="mr-3 mt-1 text-primary">•</span>
                                <span><strong>Infrastructure investment:</strong> Strategic projects that enable growth</span>
                            </li>
                        </ul>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">Regional Economic Resilience</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Beyond managing current conditions, the system helps build long-term resilience: economic diversification, skill base development, infrastructure adequacy, and institutional capacity. Resilient economies weather shocks better and recover faster.
                        </p>
                    </section>

                    <section className="rounded-lg bg-accent/10 p-8">
                        <h2 className="text-2xl font-semibold">Strengthen Your Economic Governance Capacity</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Discover how Economic Stability Intelligence can enhance your capacity to manage economic conditions proactively. Schedule a briefing on economic monitoring and intervention.
                        </p>
                        <button className="mt-6 rounded-lg bg-primary px-6 py-3 font-semibold text-primary-foreground hover:bg-primary/90">
                            Request Economic Briefing
                        </button>
                    </section>
                </div>
            </div>
        </div>
    );
}
