import { BackButton } from "@/components/back-button";

export default function PrivacyPage() {
    return (
        <main className="min-h-screen py-16 md:py-32">
            <div className="mx-auto max-w-4xl px-6">
                <BackButton />
                <h1 className="text-5xl font-bold md:text-6xl">Privacy Policy</h1>

                <div className="mt-12 space-y-8 text-muted-foreground">
                    <section>
                        <p className="text-lg">
                            Last updated: January 21, 2026
                        </p>
                    </section>

                    <section>
                        <h2 className="text-3xl font-semibold mb-4 text-foreground">Our Commitment to Privacy</h2>
                        <p className="text-lg leading-relaxed">
                            SentinelPasture is committed to protecting the privacy and security of information.
                            This privacy policy explains how we collect, use, and safeguard information when you
                            interact with our website and services.
                        </p>
                    </section>

                    <section>
                        <h2 className="text-3xl font-semibold mb-4 text-foreground">Information We Collect</h2>
                        <div className="space-y-4">
                            <div>
                                <h3 className="text-xl font-semibold mb-2 text-foreground">Website Analytics</h3>
                                <p className="leading-relaxed">
                                    We may collect basic analytics information about your visit to our website,
                                    including pages viewed, time spent, and general location data.
                                </p>
                            </div>

                            <div>
                                <h3 className="text-xl font-semibold mb-2 text-foreground">Contact Information</h3>
                                <p className="leading-relaxed">
                                    If you choose to contact us, we collect the information you provide,
                                    such as your name, email address, and message content.
                                </p>
                            </div>
                        </div>
                    </section>

                    <section>
                        <h2 className="text-3xl font-semibold mb-4 text-foreground">How We Use Information</h2>
                        <ul className="space-y-2 ml-4">
                            <li>• To respond to your inquiries and communications</li>
                            <li>• To improve our website and services</li>
                            <li>• To understand how visitors interact with our content</li>
                            <li>• To communicate about relevant updates or opportunities</li>
                        </ul>
                    </section>

                    <section>
                        <h2 className="text-3xl font-semibold mb-4 text-foreground">Data Security</h2>
                        <p className="leading-relaxed">
                            We implement appropriate technical and organizational measures to protect the
                            information we collect. However, no method of transmission over the internet
                            is 100% secure, and we cannot guarantee absolute security.
                        </p>
                    </section>

                    <section>
                        <h2 className="text-3xl font-semibold mb-4 text-foreground">Third-Party Services</h2>
                        <p className="leading-relaxed">
                            Our website may contain links to third-party sites or services. We are not
                            responsible for the privacy practices of these external sites.
                        </p>
                    </section>

                    <section>
                        <h2 className="text-3xl font-semibold mb-4 text-foreground">Contact Us</h2>
                        <p className="leading-relaxed">
                            If you have questions about this privacy policy or our data practices,
                            please contact us at{" "}
                            <a
                                href="mailto:privacy@sentinelpasture.com"
                                className="text-primary hover:underline"
                            >
                                privacy@sentinelpasture.com
                            </a>
                        </p>
                    </section>

                    <section className="mt-12 border-l-4 pl-6">
                        <p className="text-lg italic">
                            As builders of governance infrastructure, we take privacy and data protection seriously.
                            This policy reflects our commitment to transparency and responsible data handling.
                        </p>
                    </section>
                </div>
            </div>
        </main>
    );
}
