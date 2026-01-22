import { BackButton } from "@/components/back-button";

export default function Infrastructure() {
    return (
        <div className="min-h-screen">
            <div className="mx-auto max-w-4xl px-6 py-24">
                <BackButton />

                <h1 className="mt-8 text-4xl font-bold md:text-5xl">
                    Infrastructure Intelligence
                </h1>
                <p className="mt-4 text-xl text-muted-foreground">
                    Monitor critical systems before failures cascade
                </p>

                <div className="mt-12 space-y-8">
                    <section>
                        <h2 className="text-2xl font-semibold">Infrastructure as Governance Capacity</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Infrastructure isn't just physical assets—it's the foundation of your capacity to govern. When water systems fail, public health suffers. When transportation networks degrade, economic activity declines. When power grids become unreliable, everything else becomes harder.
                        </p>
                        <p className="mt-4 text-lg leading-relaxed">
                            Yet most infrastructure monitoring focuses on individual assets, missing system-level vulnerabilities and cascading failure risks. We need to understand infrastructure as interconnected systems, not isolated components.
                        </p>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">Comprehensive System Monitoring</h2>
                        <div className="mt-6 space-y-4">
                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Asset Health Tracking</h3>
                                <p className="mt-3 leading-relaxed">
                                    Condition monitoring across transportation, water, power, communications, and public facilities. Identify deterioration before it becomes failure.
                                </p>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">System Interdependencies</h3>
                                <p className="mt-3 leading-relaxed">
                                    Map how infrastructure systems depend on each other. Power failures affect water pumping. Transportation disruptions impact supply chains. Understanding dependencies is essential for managing cascading risks.
                                </p>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Capacity Utilization</h3>
                                <p className="mt-3 leading-relaxed">
                                    Monitor how close systems are to capacity limits. Overutilized infrastructure fails more frequently and degrades faster. Underutilized infrastructure represents wasted investment.
                                </p>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Resilience Assessment</h3>
                                <p className="mt-3 leading-relaxed">
                                    How well can systems withstand shocks and recover from disruptions? Identify single points of failure and insufficient redundancy before they're tested by crisis.
                                </p>
                            </div>
                        </div>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">Predictive Maintenance</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            The system predicts infrastructure failures before they occur, enabling preventive maintenance. This approach is dramatically more cost-effective than reactive repair and avoids the broader disruptions that failures cause.
                        </p>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">Strategic Investment Planning</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Infrastructure investment decisions should be based on comprehensive understanding of system condition, utilization patterns, and future demand. The system provides the analytical foundation for evidence-based infrastructure planning.
                        </p>
                    </section>

                    <section className="rounded-lg bg-accent/10 p-8">
                        <h2 className="text-2xl font-semibold">Protect Your Infrastructure Foundation</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Discover how Infrastructure Intelligence can enhance your capacity to manage critical systems. Schedule a technical consultation.
                        </p>
                        <button className="mt-6 rounded-lg bg-primary px-6 py-3 font-semibold text-primary-foreground hover:bg-primary/90">
                            Request Infrastructure Consultation
                        </button>
                    </section>
                </div>
            </div>
        </div>
    );
}
