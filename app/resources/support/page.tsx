import { BackButton } from "@/components/back-button";

export default function Support() {
    return (
        <div className="min-h-screen">
            <div className="mx-auto max-w-4xl px-6 py-24">
                <BackButton />

                <h1 className="mt-8 text-4xl font-bold md:text-5xl">
                    Support Center
                </h1>
                <p className="mt-4 text-xl text-muted-foreground">
                    Training, technical support, and implementation assistance
                </p>

                <div className="mt-12 space-y-8">
                    <section>
                        <h2 className="text-2xl font-semibold">Comprehensive Support Services</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Successful implementation of governance intelligence systems requires more than technology—it requires organizational change, capability development, and ongoing support. We provide comprehensive assistance throughout your journey.
                        </p>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">Support Services</h2>
                        <div className="mt-6 space-y-6">
                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Implementation Consulting</h3>
                                <p className="mt-3 leading-relaxed">
                                    Expert guidance on deployment planning, data integration, organizational change management, and capability development.
                                </p>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Training Programs</h3>
                                <p className="mt-3 leading-relaxed">
                                    Role-specific training for executives, analysts, technical staff, and operational users. From strategic overview to technical deep-dives.
                                </p>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Technical Support</h3>
                                <p className="mt-3 leading-relaxed">
                                    24/7 technical assistance for system operation, troubleshooting, and optimization. Multiple support tiers based on urgency and complexity.
                                </p>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Analytical Consulting</h3>
                                <p className="mt-3 leading-relaxed">
                                    Expert analysts available to assist with complex analyses, custom model development, and interpretation of findings.
                                </p>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Continuous Improvement</h3>
                                <p className="mt-3 leading-relaxed">
                                    Regular system reviews, performance optimization, and capability enhancement as your needs evolve.
                                </p>
                            </div>
                        </div>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">Support Channels</h2>
                        <ul className="mt-4 space-y-3 text-lg">
                            <li className="flex items-start">
                                <span className="mr-3 mt-1 text-primary">•</span>
                                <span><strong>Support Portal:</strong> Submit tickets, track issues, access knowledge base</span>
                            </li>
                            <li className="flex items-start">
                                <span className="mr-3 mt-1 text-primary">•</span>
                                <span><strong>Direct Contact:</strong> Phone and email support for urgent issues</span>
                            </li>
                            <li className="flex items-start">
                                <span className="mr-3 mt-1 text-primary">•</span>
                                <span><strong>Scheduled Consultations:</strong> Regular check-ins with your support team</span>
                            </li>
                            <li className="flex items-start">
                                <span className="mr-3 mt-1 text-primary">•</span>
                                <span><strong>User Community:</strong> Connect with other jurisdictions using the platform</span>
                            </li>
                        </ul>
                    </section>

                    <section className="rounded-lg bg-accent/10 p-8">
                        <h2 className="text-2xl font-semibold">Get Support</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Contact our support team to discuss your needs or access the support portal.
                        </p>
                        <div className="mt-6 flex gap-4">
                            <button className="rounded-lg bg-primary px-6 py-3 font-semibold text-primary-foreground hover:bg-primary/90">
                                Contact Support
                            </button>
                            <button className="rounded-lg border px-6 py-3 font-semibold hover:bg-accent">
                                Access Support Portal
                            </button>
                        </div>
                    </section>
                </div>
            </div>
        </div>
    );
}
