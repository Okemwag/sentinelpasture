import { BackButton } from "@/components/back-button";

export default function Documentation() {
    return (
        <div className="min-h-screen">
            <div className="mx-auto max-w-4xl px-6 py-24">
                <BackButton />

                <h1 className="mt-8 text-4xl font-bold md:text-5xl">
                    Documentation
                </h1>
                <p className="mt-4 text-xl text-muted-foreground">
                    Technical guides, API documentation, and implementation resources
                </p>

                <div className="mt-12 space-y-8">
                    <section>
                        <h2 className="text-2xl font-semibold">Comprehensive Technical Documentation</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Our documentation is designed for technical teams implementing and operating the governance intelligence platform. From initial setup to advanced customization, we provide the guidance you need.
                        </p>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">Documentation Categories</h2>
                        <div className="mt-6 grid gap-6 md:grid-cols-2">
                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Getting Started</h3>
                                <p className="mt-3 leading-relaxed">
                                    Installation guides, system requirements, initial configuration, and quick-start tutorials.
                                </p>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Platform Architecture</h3>
                                <p className="mt-3 leading-relaxed">
                                    System design, component interactions, data flows, and scalability considerations.
                                </p>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Data Integration</h3>
                                <p className="mt-3 leading-relaxed">
                                    Connector configuration, data mapping, quality assurance, and synchronization protocols.
                                </p>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">API Reference</h3>
                                <p className="mt-3 leading-relaxed">
                                    Complete API documentation with endpoints, parameters, authentication, and code examples.
                                </p>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Security and Compliance</h3>
                                <p className="mt-3 leading-relaxed">
                                    Access control configuration, data protection, audit logging, and regulatory compliance.
                                </p>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Customization Guide</h3>
                                <p className="mt-3 leading-relaxed">
                                    Custom analytics, dashboard configuration, alert rules, and workflow automation.
                                </p>
                            </div>
                        </div>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">Support Resources</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Beyond documentation, we provide technical support, training programs, and implementation consulting to ensure successful deployment and operation.
                        </p>
                    </section>

                    <section className="rounded-lg bg-accent/10 p-8">
                        <h2 className="text-2xl font-semibold">Access Technical Documentation</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Request access to our comprehensive technical documentation portal. Available to authorized implementation partners and client technical teams.
                        </p>
                        <button className="mt-6 rounded-lg bg-primary px-6 py-3 font-semibold text-primary-foreground hover:bg-primary/90">
                            Request Documentation Access
                        </button>
                    </section>
                </div>
            </div>
        </div>
    );
}
