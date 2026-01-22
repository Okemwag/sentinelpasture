import { BackButton } from "@/components/back-button";

export default function NewsroomPage() {
    return (
        <main className="min-h-screen py-16 md:py-32">
            <div className="mx-auto max-w-4xl px-6">
                <BackButton />
                <h1 className="text-5xl font-bold md:text-6xl">Newsroom</h1>

                <div className="mt-12 space-y-8">
                    <section>
                        <p className="text-xl text-muted-foreground">
                            Research, deployments, and public documentation of our work.
                        </p>
                    </section>

                    <section className="mt-16">
                        <h2 className="text-3xl font-semibold mb-6">Latest Updates</h2>
                        <div className="space-y-6">
                            <div className="border-b pb-6">
                                <p className="text-sm text-muted-foreground mb-2">Coming Soon</p>
                                <h3 className="text-xl font-semibold mb-2">Research Publications</h3>
                                <p className="text-muted-foreground">
                                    Technical papers and research findings on governance intelligence,
                                    early warning systems, and institutional resilience.
                                </p>
                            </div>

                            <div className="border-b pb-6">
                                <p className="text-sm text-muted-foreground mb-2">Coming Soon</p>
                                <h3 className="text-xl font-semibold mb-2">Deployment Case Studies</h3>
                                <p className="text-muted-foreground">
                                    Documented learnings from real-world deployments, including challenges
                                    encountered and solutions developed.
                                </p>
                            </div>

                            <div className="border-b pb-6">
                                <p className="text-sm text-muted-foreground mb-2">Coming Soon</p>
                                <h3 className="text-xl font-semibold mb-2">Technical Documentation</h3>
                                <p className="text-muted-foreground">
                                    Public documentation of our methodologies, frameworks, and approaches
                                    to building governance infrastructure.
                                </p>
                            </div>
                        </div>
                    </section>

                    <section className="mt-16 bg-muted/50 rounded-lg p-8">
                        <h2 className="text-2xl font-semibold mb-4">Transparency & Accountability</h2>
                        <p className="text-muted-foreground leading-relaxed">
                            We believe that governance infrastructure must be transparent and accountable.
                            This newsroom serves as a public record of our work, research, and deployments.
                            We are committed to documenting our methods, sharing our learnings, and
                            maintaining public trust through openness about what we build and how it&apos;s used.
                        </p>
                    </section>

                    <section className="mt-12 border-l-4 pl-6">
                        <p className="text-lg italic text-muted-foreground">
                            Banditry is the first proof. National resilience is the mission.
                        </p>
                    </section>
                </div>
            </div>
        </main>
    );
}
