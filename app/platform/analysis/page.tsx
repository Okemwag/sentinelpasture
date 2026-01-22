import { BackButton } from "@/components/back-button";

export default function AnalysisEngine() {
    return (
        <div className="min-h-screen">
            <div className="mx-auto max-w-4xl px-6 py-24">
                <BackButton />

                <h1 className="mt-8 text-4xl font-bold md:text-5xl">
                    Analysis Engine
                </h1>
                <p className="mt-4 text-xl text-muted-foreground">
                    Deep causal analysis for evidence-based decision-making
                </p>

                <div className="mt-12 space-y-8">
                    <section>
                        <h2 className="text-2xl font-semibold">Beyond Correlation: Understanding Causation</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Effective governance requires understanding not just what is happening, but why it's happening. Correlation tells you that two things move together. Causation tells you whether changing one will affect the other. This distinction is the difference between informed intervention and expensive guesswork.
                        </p>
                        <p className="mt-4 text-lg leading-relaxed">
                            The Analysis Engine applies advanced causal inference methods to governance data, helping you understand the actual drivers of conditions in your jurisdiction. This is the foundation of evidence-based policy.
                        </p>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">The Questions Leaders Need Answered</h2>
                        <div className="mt-6 space-y-6">
                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">What's Actually Causing This?</h3>
                                <p className="mt-3 leading-relaxed">
                                    When crime rises in a district, is it driven by economic stress, demographic shifts, reduced police presence, or something else entirely? The engine analyzes multiple potential causes simultaneously, quantifying the contribution of each factor.
                                </p>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">What Will Happen If We Do Nothing?</h3>
                                <p className="mt-3 leading-relaxed">
                                    Scenario modeling based on historical patterns and current trajectories. Understand the likely evolution of situations under status quo conditions. This baseline is essential for evaluating intervention options.
                                </p>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">What Will Happen If We Intervene?</h3>
                                <p className="mt-3 leading-relaxed">
                                    Counterfactual analysis: estimate the likely effects of different interventions based on historical evidence and causal models. Compare options before committing resources.
                                </p>
                            </div>

                            <div className="rounded-lg border p-6">
                                <h3 className="text-xl font-semibold">Are There Unintended Consequences?</h3>
                                <p className="mt-3 leading-relaxed">
                                    The engine identifies potential spillover effects and second-order impacts. Interventions in one domain often affect others. See the full picture before you act.
                                </p>
                            </div>
                        </div>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">Rigorous Methods, Accessible Insights</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            The Analysis Engine employs state-of-the-art causal inference techniques—difference-in-differences, synthetic controls, regression discontinuity, instrumental variables—but presents findings in language that supports decision-making, not academic publication.
                        </p>
                        <p className="mt-4 text-lg leading-relaxed">
                            Every analysis includes:
                        </p>
                        <ul className="mt-4 space-y-3 text-lg">
                            <li className="flex items-start">
                                <span className="mr-3 mt-1 text-primary">•</span>
                                <span><strong>Plain-language summary:</strong> What we found and why it matters</span>
                            </li>
                            <li className="flex items-start">
                                <span className="mr-3 mt-1 text-primary">•</span>
                                <span><strong>Confidence assessment:</strong> How certain we are about each finding</span>
                            </li>
                            <li className="flex items-start">
                                <span className="mr-3 mt-1 text-primary">•</span>
                                <span><strong>Policy implications:</strong> What this means for decision-making</span>
                            </li>
                            <li className="flex items-start">
                                <span className="mr-3 mt-1 text-primary">•</span>
                                <span><strong>Methodological transparency:</strong> How we reached these conclusions</span>
                            </li>
                        </ul>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">Learning from Your Own Experience</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            The engine doesn't just analyze current situations—it learns from your jurisdiction's history. Every intervention becomes a natural experiment. The system tracks outcomes, compares them to predictions, and refines its models continuously.
                        </p>
                        <p className="mt-4 text-lg leading-relaxed">
                            Over time, you build an evidence base specific to your context. This is how governance becomes genuinely evidence-based: by learning systematically from experience.
                        </p>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold">Real-World Application</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            A regional government used the Analysis Engine to understand rising youth unemployment. Traditional analysis suggested economic factors. The engine revealed that transportation access was the primary constraint—most job opportunities were geographically inaccessible to affected populations.
                        </p>
                        <p className="mt-4 text-lg leading-relaxed">
                            The intervention shifted from job training programs (addressing a symptom) to targeted transportation subsidies (addressing the cause). Youth employment rose 34% within six months. This is the power of causal understanding.
                        </p>
                    </section>

                    <section className="rounded-lg bg-accent/10 p-8">
                        <h2 className="text-2xl font-semibold">Make Decisions Based on Evidence, Not Assumptions</h2>
                        <p className="mt-4 text-lg leading-relaxed">
                            Discover how the Analysis Engine can strengthen your evidence base for policy decisions. Schedule a technical briefing tailored to your analytical needs.
                        </p>
                        <button className="mt-6 rounded-lg bg-primary px-6 py-3 font-semibold text-primary-foreground hover:bg-primary/90">
                            Request Technical Briefing
                        </button>
                    </section>
                </div>
            </div>
        </div>
    );
}
