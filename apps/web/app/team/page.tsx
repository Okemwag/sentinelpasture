import { BackButton } from "@/components/back-button";

export default function TeamPage() {
    return (
        <main className="min-h-screen py-16 md:py-32">
            <div className="mx-auto max-w-4xl px-6">
                <BackButton />
                <h1 className="text-5xl font-bold md:text-6xl">Team</h1>

                <div className="mt-12 space-y-8">
                    <section>
                        <p className="text-xl text-muted-foreground leading-relaxed">
                            Engineers, policy thinkers, and practitioners working at the intersection
                            of AI, governance, and public trust.
                        </p>
                    </section>

                    <section className="mt-16">
                        <h2 className="text-3xl font-semibold mb-6">Our Expertise</h2>
                        <div className="grid gap-6 md:grid-cols-2">
                            <div className="border rounded-lg p-6">
                                <h3 className="text-xl font-semibold mb-2">Engineering</h3>
                                <p className="text-muted-foreground">
                                    Building robust, scalable systems that can process and analyze complex signals
                                    from multiple sources in real-time.
                                </p>
                            </div>

                            <div className="border rounded-lg p-6">
                                <h3 className="text-xl font-semibold mb-2">Policy & Governance</h3>
                                <p className="text-muted-foreground">
                                    Deep understanding of institutional structures, decision-making processes,
                                    and governance frameworks.
                                </p>
                            </div>

                            <div className="border rounded-lg p-6">
                                <h3 className="text-xl font-semibold mb-2">AI & Machine Learning</h3>
                                <p className="text-muted-foreground">
                                    Advanced capabilities in signal fusion, pattern recognition, and
                                    early warning systems.
                                </p>
                            </div>

                            <div className="border rounded-lg p-6">
                                <h3 className="text-xl font-semibold mb-2">Public Trust</h3>
                                <p className="text-muted-foreground">
                                    Commitment to transparency, accountability, and ethical deployment
                                    of governance technology.
                                </p>
                            </div>
                        </div>
                    </section>

                    <section className="mt-16 border-l-4 pl-6">
                        <p className="text-lg italic text-muted-foreground">
                            We are building systems meant to hold under pressure.
                            That requires people who take responsibility seriously.
                        </p>
                    </section>
                </div>
            </div>
        </main>
    );
}
