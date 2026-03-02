import Image from "next/image";

export default function PlatformVisual() {
    return (
        <section className="py-16 md:py-32 bg-background">
            <div className="mx-auto max-w-6xl px-6">
                <div className="text-center mb-12">
                    <h2 className="text-4xl font-bold lg:text-5xl mb-4">
                        Platform Architecture
                    </h2>
                    <p className="text-lg text-muted-foreground max-w-3xl mx-auto">
                        A comprehensive system for multi-signal intelligence, early warning detection,
                        and coordinated governance response.
                    </p>
                </div>

                <div className="grid gap-8 md:grid-cols-3">
                    <div className="flex flex-col items-center text-center space-y-4 p-6 rounded-xl border bg-card hover:shadow-lg transition-shadow">
                        <div className="relative w-full aspect-square max-w-xs">
                            <Image
                                src="/governance-network.png"
                                alt="Multi-Signal Intelligence Network"
                                width={400}
                                height={400}
                                className="rounded-lg object-contain"
                            />
                        </div>
                        <h3 className="text-xl font-semibold">Signal Fusion</h3>
                        <p className="text-sm text-muted-foreground">
                            Integrating security, economic, social, climate, and community signals
                            into unified operational intelligence.
                        </p>
                    </div>

                    <div className="flex flex-col items-center text-center space-y-4 p-6 rounded-xl border bg-card hover:shadow-lg transition-shadow">
                        <div className="relative w-full aspect-square max-w-xs bg-muted/50 rounded-lg flex items-center justify-center p-4">
                            <Image
                                src="/decision-infrastructure.png"
                                alt="Decision Infrastructure Hierarchy"
                                width={400}
                                height={400}
                                className="rounded-lg object-contain"
                            />
                        </div>
                        <h3 className="text-xl font-semibold">Decision Infrastructure</h3>
                        <p className="text-sm text-muted-foreground">
                            Hierarchical coordination pathways that aggregate warnings and
                            route them to appropriate decision-makers.
                        </p>
                    </div>

                    <div className="flex flex-col items-center text-center space-y-4 p-6 rounded-xl border bg-card hover:shadow-lg transition-shadow">
                        <div className="relative w-full aspect-square max-w-xs bg-muted/50 rounded-lg flex items-center justify-center p-4">
                            <Image
                                src="/early-warning.png"
                                alt="Early Warning Detection System"
                                width={400}
                                height={400}
                                className="rounded-lg object-contain"
                            />
                        </div>
                        <h3 className="text-xl font-semibold">Early Warning</h3>
                        <p className="text-sm text-muted-foreground">
                            Concentric detection layers that identify emerging instability
                            before it escalates into crisis.
                        </p>
                    </div>
                </div>
            </div>
        </section>
    );
}
