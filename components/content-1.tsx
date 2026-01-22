import Image from "next/image";

export default function ContentSection() {
  return (
    <section className="py-16 md:py-32">
      <div className="mx-auto max-w-5xl space-y-8 px-6 md:space-y-16">
        <h2 className="relative z-10 max-w-xl text-4xl font-medium lg:text-5xl">
          Not surveillance. Not predictive policing. Not automation of authority.
        </h2>
        <div className="grid gap-6 sm:grid-cols-2 md:gap-12 lg:gap-24">
          <div className="relative mb-6 sm:mb-0">
            <div className="bg-linear-to-b aspect-square relative rounded-2xl from-zinc-300 to-transparent p-px dark:from-zinc-700 flex items-center justify-center bg-muted/50">
              <Image
                src="/decision-infrastructure.png"
                className="rounded-[15px] p-8 object-contain"
                alt="governance decision infrastructure"
                width={1207}
                height={929}
              />
            </div>
          </div>

          <div className="relative space-y-4">
            <p className="text-muted-foreground">
              This is{" "}
              <span className="text-accent-foreground font-bold">
                decision infrastructure for modern governance.
              </span>
            </p>
            <p className="text-muted-foreground">
              Where governments today react to incidents, this system enables them to govern pressure, allocate response intelligently, and preserve resilience across regions and sectors.
            </p>
            <p className="text-muted-foreground">
              We build core governance infrastructure for early warning, coordination, and institutional learning.
            </p>

            <div className="pt-6">
              <blockquote className="border-l-4 pl-4">
                <p className="text-muted-foreground italic">
                  Banditry is the first proof. National resilience is the mission.
                </p>
              </blockquote>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
