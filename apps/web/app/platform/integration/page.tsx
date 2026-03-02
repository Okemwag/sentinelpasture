import { BackButton } from "@/components/back-button";

export default function DataIntegration() {
    return (
        <div className="min-h-screen">
            <div className="mx-auto max-w-4xl px-6 py-24">
                <BackButton />

                <h1 className="mt-8 text-4xl font-bold md:text-5xl">
                    Data Integration
                </h1>
                <p className="mt-4 text-xl text-muted-foreground">
                    Connect your data sources into unified governance intelligence
                </p>

                <div className="mt-12 space-y-8">
                    <section>
                        <h2 className="text-2xl font-semibold">The Data Fragmentation Problem</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Most governments possess vast amounts of data—economic statistics, public safety records, infrastructure sensors, social services data, environmental monitoring, health surveillance. But this data exists in isolated systems, incompatible formats, and organizational silos.
                        </p>
                        <p className="mt-4 text-lg leading-relaxed">
                            The result: you have data but lack intelligence. Patterns that span multiple domains remain invisible. Connections between seemingly unrelated issues go undetected. Decision-makers work with incomplete pictures because the complete picture doesn't exist anywhere.
                        </p>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">Integration Without Disruption</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Data Integration connects your existing systems without requiring wholesale replacement or reorganization. We work with what you have, creating a unified intelligence layer above your current infrastructure.
                        </p>

                        <div className="mt-6 space-y-6">
                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Flexible Connectivity</h3>
                                <p className="mt-3 leading-relaxed">
                                    Connect to databases, APIs, file systems, IoT sensors, and legacy systems. We support standard protocols and can develop custom connectors for specialized systems. Your data stays where it is—we bring it together virtually.
                                </p>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Automated Harmonization</h3>
                                <p className="mt-3 leading-relaxed">
                                    Different systems use different formats, schemas, and conventions. Our integration layer automatically harmonizes data into consistent formats, resolves naming conflicts, and aligns temporal and spatial references.
                                </p>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Real-Time Synchronization</h3>
                                <p className="mt-3 leading-relaxed">
                                    As source systems update, the integrated view updates automatically. No manual data transfers, no batch processing delays. Intelligence is always current.
                                </p>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Quality Assurance</h3>
                                <p className="mt-3 leading-relaxed">
                                    Continuous monitoring for data quality issues—missing values, outliers, inconsistencies, staleness. Problems are flagged immediately, with impact assessment on downstream intelligence.
                                </p>
                            </div>
                        </div>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">Security and Governance</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Integration doesn't mean unrestricted access. The system enforces fine-grained access controls, ensuring that users see only data they're authorized to access. Audit logs track every data access for accountability.
                        </p>
                        <ul className="mt-4 space-y-3 text-lg">
                            <li className="flex items-start">
                                <span className="mr-3 mt-1 text-primary">•</span>
                                <span><strong>Role-based access control:</strong> Permissions aligned with organizational roles</span>
                            </li>
                            <li className="flex items-start">
                                <span className="mr-3 mt-1 text-primary">•</span>
                                <span><strong>Data classification:</strong> Sensitive data protected with additional controls</span>
                            </li>
                            <li className="flex items-start">
                                <span className="mr-3 mt-1 text-primary">•</span>
                                <span><strong>Privacy preservation:</strong> Automated anonymization and aggregation where required</span>
                            </li>
                            <li className="flex items-start">
                                <span className="mr-3 mt-1 text-primary">•</span>
                                <span><strong>Compliance enforcement:</strong> Built-in controls for regulatory requirements</span>
                            </li>
                        </ul>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">From Data to Intelligence</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Integration is the foundation, not the destination. Once data is connected, the platform's analytical capabilities can work across domains:
                        </p>
                        <ul className="mt-4 space-y-3 text-lg">
                            <li className="flex items-start">
                                <span className="mr-3 mt-1 text-primary">•</span>
                                <span>Detect patterns that span multiple data sources</span>
                            </li>
                            <li className="flex items-start">
                                <span className="mr-3 mt-1 text-primary">•</span>
                                <span>Understand causal relationships between different domains</span>
                            </li>
                            <li className="flex items-start">
                                <span className="mr-3 mt-1 text-primary">•</span>
                                <span>Identify early warning signals from cross-domain indicators</span>
                            </li>
                            <li className="flex items-start">
                                <span className="mr-3 mt-1 text-primary">•</span>
                                <span>Generate comprehensive situational awareness</span>
                            </li>
                        </ul>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">Incremental Implementation</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            You don't need to integrate everything at once. Start with high-priority data sources, demonstrate value, then expand. Each new connection increases the intelligence available to decision-makers.
                        </p>
                        <p className="mt-4 text-lg leading-relaxed">
                            We work with your IT teams to plan integration sequencing, manage technical dependencies, and ensure smooth deployment. This is a partnership, not a product delivery.
                        </p>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">Technical Excellence, Operational Focus</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            The integration layer is technically sophisticated—handling schema mapping, data quality, real-time synchronization, and security—but operationally transparent. Your teams use the intelligence, not the integration infrastructure.
                        </p>
                        <p className="mt-4 text-lg leading-relaxed">
                            This is how technology should work: complex capabilities, simple experience.
                        </p>
                    </section>

                    <section className="rounded-lg bg-accent/10 p-8">
                        <h2 className="text-2xl font-semibold">Turn Data Silos into Unified Intelligence</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Discuss your data integration challenges with our technical team. We'll assess your current landscape and design an integration approach tailored to your environment.
                        </p>
                        <button className="mt-6 rounded-lg bg-primary px-6 py-3 font-semibold text-primary-foreground hover:bg-primary/90">
                            Request Integration Assessment
                        </button>
                    </section>
                </div>
            </div>
        </div>
    );
}
