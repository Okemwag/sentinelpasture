import { BackButton } from "@/components/back-button";

export default function AboutPage() {
    return (
        <main className="min-h-screen py-16 md:py-32">
            <div className="mx-auto max-w-4xl px-6">
                <BackButton />
                <h1 className="text-5xl font-bold md:text-6xl">About</h1>

                <div className="mt-12 space-y-8">
                    <section>
                        <h2 className="text-3xl font-semibold mb-4">Our Mission</h2>
                        <p className="text-lg text-muted-foreground">
                            We build core governance infrastructure for early warning, coordination, and institutional learning.
                        </p>
                    </section>

                    <section>
                        <h2 className="text-3xl font-semibold mb-4">What We Do</h2>
                        <p className="text-lg text-muted-foreground mb-4">
                            We are building a governance intelligence system for detecting instability early,
                            understanding why it is emerging, and coordinating proportionate response before crises escalate.
                        </p>
                        <p className="text-lg text-muted-foreground">
                            The platform fuses security, economic, social, climate, and community signals into a
                            single operational understanding of societal risk. Not to predict violence — but to prevent instability.
                        </p>
                    </section>

                    <section>
                        <h2 className="text-3xl font-semibold mb-4">Our Approach</h2>
                        <p className="text-lg text-muted-foreground mb-4">
                            Where governments today react to incidents, this system enables them to govern pressure,
                            allocate response intelligently, and preserve resilience across regions and sectors.
                        </p>
                        <div className="border-l-4 pl-6 my-6">
                            <p className="text-xl font-medium">This is not surveillance.</p>
                            <p className="text-xl font-medium">This is not predictive policing.</p>
                            <p className="text-xl font-medium">This is not automation of authority.</p>
                            <p className="text-xl font-bold mt-4">It is decision infrastructure for modern governance.</p>
                        </div>
                    </section>

                    <section>
                        <h2 className="text-3xl font-semibold mb-4">Our Focus</h2>
                        <p className="text-lg text-muted-foreground">
                            Banditry is the first proof. National resilience is the mission.
                        </p>
                    </section>
                </div>
            </div>
        </main>
    );
}
