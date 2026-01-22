import { BackButton } from "@/components/back-button";

export default function CaseStudies() {
    return (
        <div className="min-h-screen">
            <div className="mx-auto max-w-4xl px-6 py-24">
                <BackButton />

                <h1 className="mt-8 text-4xl font-bold md:text-5xl">
                    Case Studies
                </h1>
                <p className="mt-4 text-xl text-muted-foreground">
                    Real-world applications of governance intelligence
                </p>

                <div className="mt-12 space-y-8">
                    <section>
                        <h2 className="text-2xl font-semibold">Learning from Implementation</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Our case studies document how regional bodies and state institutions have implemented governance intelligence systems to address specific challenges. These are not marketing materials—they're honest accounts of implementation, including challenges encountered and lessons learned.
                        </p>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">Featured Case Studies</h2>
                        <div className="mt-6 space-y-6">
                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Regional Public Safety Transformation</h3>
                                <p className="mt-3 text-sm text-muted-foreground">Metropolitan Region | 18-month implementation</p>
                                <p className="mt-3 leading-relaxed">
                                    How a regional government shifted from reactive policing to proactive safety governance, reducing crime rates by 25% while improving community confidence and equity in service delivery.
                                </p>
                                <button className="mt-4 text-primary hover:underline">Read Full Case Study →</button>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Economic Resilience During Crisis</h3>
                                <p className="mt-3 text-sm text-muted-foreground">State Government | Crisis response</p>
                                <p className="mt-3 leading-relaxed">
                                    Using early warning systems to detect economic deterioration 18 days earlier than traditional indicators, enabling targeted interventions that prevented widespread business failures.
                                </p>
                                <button className="mt-4 text-primary hover:underline">Read Full Case Study →</button>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Infrastructure Failure Prevention</h3>
                                <p className="mt-3 text-sm text-muted-foreground">Regional Authority | Ongoing program</p>
                                <p className="mt-3 leading-relaxed">
                                    Predictive maintenance program that reduced infrastructure failures by 60% and maintenance costs by 40% through early detection and preventive intervention.
                                </p>
                                <button className="mt-4 text-primary hover:underline">Read Full Case Study →</button>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Cross-Jurisdictional Coordination</h3>
                                <p className="mt-3 text-sm text-muted-foreground">Multi-state initiative | 24-month implementation</p>
                                <p className="mt-3 leading-relaxed">
                                    How three neighboring states implemented shared intelligence infrastructure to coordinate responses to challenges spanning jurisdictional boundaries.
                                </p>
                                <button className="mt-4 text-primary hover:underline">Read Full Case Study →</button>
                            </div>
                        </div>
                    </section>

                    <section className="rounded-lg bg-accent/10 p-8">
                        <h2 className="text-2xl font-semibold">Explore Detailed Case Studies</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Request access to our complete case study library, including implementation details, technical specifications, and outcome measurements.
                        </p>
                        <button className="mt-6 rounded-lg bg-primary px-6 py-3 font-semibold text-primary-foreground hover:bg-primary/90">
                            Request Case Study Access
                        </button>
                    </section>
                </div>
            </div>
        </div>
    );
}
