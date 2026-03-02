import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { AlertTriangle, TrendingDown, Users } from "lucide-react";

const caseStudies = [
    {
        title: "Rwanda 1994",
        subtitle: "The Cost of Ignored Early Warnings",
        icon: AlertTriangle,
        impact: "800,000+ lives lost",
        description:
            "Multiple intelligence reports and warnings from UN peacekeepers were ignored. The international community failed to act on clear signals of impending genocide. A governance intelligence system could have aggregated these warnings, escalated them appropriately, and coordinated early intervention.",
        lesson: "Early warning without coordination is insufficient.",
    },
    {
        title: "2008 Financial Crisis",
        subtitle: "Failure to Detect Systemic Risk",
        icon: TrendingDown,
        impact: "$10+ trillion in economic losses",
        description:
            "Economic signals were fragmented across institutions. No single system integrated housing market data, lending patterns, and systemic risk indicators. The crisis escalated because decision-makers lacked a unified operational understanding of emerging instability.",
        lesson: "Siloed data creates blind spots in governance.",
    },
    {
        title: "Arab Spring",
        subtitle: "Inability to Understand Emerging Instability",
        icon: Users,
        impact: "Regional destabilization, ongoing conflicts",
        description:
            "Governments missed the convergence of economic pressure, social media mobilization, and political grievances. Traditional intelligence focused on security threats, not the fusion of economic, social, and climate signals that predicted widespread uprisings.",
        lesson: "Modern instability requires multi-signal intelligence.",
    },
];

export default function CaseStudies() {
    return (
        <section className="py-16 md:py-32 bg-muted/30">
            <div className="mx-auto max-w-6xl px-6">
                <div className="text-center mb-12">
                    <h2 className="text-4xl font-bold lg:text-5xl mb-4">
                        The Cost of Reactive Governance
                    </h2>
                    <p className="text-lg text-muted-foreground max-w-3xl mx-auto">
                        History shows us that governing events instead of conditions leads to
                        preventable crises. These are not just failures — they are the reasons
                        we exist.
                    </p>
                </div>

                <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-3">
                    {caseStudies.map((study, index) => {
                        const Icon = study.icon;
                        return (
                            <Card
                                key={index}
                                className="group hover:shadow-xl transition-all duration-300 border-2 hover:border-primary/50"
                            >
                                <CardHeader className="space-y-4">
                                    <div className="flex items-start justify-between">
                                        <div className="p-3 rounded-lg bg-primary/10 group-hover:bg-primary/20 transition-colors">
                                            <Icon className="size-6 text-primary" />
                                        </div>
                                        <span className="text-xs font-semibold text-muted-foreground bg-muted px-3 py-1 rounded-full">
                                            {study.impact}
                                        </span>
                                    </div>
                                    <div>
                                        <h3 className="text-2xl font-bold">{study.title}</h3>
                                        <p className="text-sm text-muted-foreground mt-1">
                                            {study.subtitle}
                                        </p>
                                    </div>
                                </CardHeader>

                                <CardContent className="space-y-4">
                                    <p className="text-sm leading-relaxed text-muted-foreground">
                                        {study.description}
                                    </p>
                                    <div className="pt-4 border-t">
                                        <p className="text-sm font-semibold italic">
                                            "{study.lesson}"
                                        </p>
                                    </div>
                                </CardContent>
                            </Card>
                        );
                    })}
                </div>

                <div className="mt-16 max-w-3xl mx-auto text-center border-l-4 border-primary pl-6">
                    <p className="text-xl font-medium">
                        Where governments today react to incidents, SentinelPasture enables them
                        to govern pressure, allocate response intelligently, and preserve
                        resilience before crises escalate.
                    </p>
                </div>
            </div>
        </section>
    );
}
