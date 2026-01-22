import { Button } from "@/components/ui/button";
import Link from "next/link";
import { BackButton } from "@/components/back-button";

export default function CareersPage() {
    return (
        <main className="min-h-screen py-16 md:py-32">
            <div className="mx-auto max-w-4xl px-6">
                <BackButton />
                <h1 className="text-5xl font-bold md:text-6xl">Careers</h1>

                <div className="mt-12 space-y-8">
                    <section className="border-l-4 pl-6">
                        <p className="text-2xl font-medium leading-relaxed">
                            We are building systems meant to hold under pressure.
                            That requires people who take responsibility seriously.
                        </p>
                    </section>

                    <section className="mt-16">
                        <h2 className="text-3xl font-semibold mb-6">What We Look For</h2>
                        <div className="space-y-6">
                            <div>
                                <h3 className="text-xl font-semibold mb-2">Deep Technical Excellence</h3>
                                <p className="text-muted-foreground">
                                    You understand that governance infrastructure must be reliable,
                                    secure, and built to withstand scrutiny.
                                </p>
                            </div>

                            <div>
                                <h3 className="text-xl font-semibold mb-2">Systems Thinking</h3>
                                <p className="text-muted-foreground">
                                    You see connections between security, economics, social dynamics,
                                    and climate — and understand how they interact.
                                </p>
                            </div>

                            <div>
                                <h3 className="text-xl font-semibold mb-2">Institutional Understanding</h3>
                                <p className="text-muted-foreground">
                                    You recognize that technology alone doesn&apos;t solve governance challenges.
                                    Context, trust, and human decision-making matter.
                                </p>
                            </div>

                            <div>
                                <h3 className="text-xl font-semibold mb-2">Responsibility & Ethics</h3>
                                <p className="text-muted-foreground">
                                    You understand the weight of building systems that inform critical decisions.
                                    You take this responsibility seriously.
                                </p>
                            </div>
                        </div>
                    </section>

                    <section className="mt-16">
                        <h2 className="text-3xl font-semibold mb-6">Our Values</h2>
                        <ul className="space-y-3 text-lg text-muted-foreground">
                            <li>• Build for resilience, not just efficiency</li>
                            <li>• Prioritize institutional learning over automation</li>
                            <li>• Earn and maintain public trust</li>
                            <li>• Work at the intersection of technology and governance</li>
                            <li>• Take responsibility for the systems we build</li>
                        </ul>
                    </section>

                    <section className="mt-16 bg-muted/50 rounded-lg p-8">
                        <h2 className="text-2xl font-semibold mb-4">Interested in Joining Us?</h2>
                        <p className="text-muted-foreground mb-6">
                            We&apos;re looking for exceptional individuals who want to work on systems
                            that matter. If you&apos;re interested in contributing to governance infrastructure
                            that can prevent crises and preserve resilience, we&apos;d like to hear from you.
                        </p>
                        <Button asChild size="lg">
                            <Link href="mailto:careers@sentinelpasture.com">
                                <span>Get in Touch</span>
                            </Link>
                        </Button>
                    </section>
                </div>
            </div>
        </main>
    );
}
