export default function GovernanceGuardrails() {
  return (
    <section className="bg-muted/30 py-16 md:py-24">
      <div className="mx-auto grid max-w-5xl gap-10 px-6 md:grid-cols-2">
        <div className="space-y-4">
          <p className="text-sm font-semibold uppercase tracking-[0.2em] text-muted-foreground">
            What It Is
          </p>
          <h2 className="text-3xl font-semibold lg:text-4xl">
            Early warning and coordination for public institutions.
          </h2>
          <p className="text-muted-foreground">
            The system is designed to help national and county teams detect
            pressure early, understand its drivers, and coordinate proportionate
            response before conditions escalate.
          </p>
          <ul className="space-y-3 text-sm text-muted-foreground">
            <li>County-level and regional views, not person-level monitoring.</li>
            <li>Aggregated indicators only, with documented constraints.</li>
            <li>Human-reviewed decisions with auditable records.</li>
          </ul>
        </div>

        <div className="space-y-4">
          <p className="text-sm font-semibold uppercase tracking-[0.2em] text-muted-foreground">
            What It Is Not
          </p>
          <div className="rounded-2xl border bg-background p-6">
            <ul className="space-y-3 text-sm text-muted-foreground">
              <li>No surveillance of individuals or household-level tracking.</li>
              <li>No predictive policing or automated enforcement.</li>
              <li>No individual targeting, watchlists, or coercive scoring.</li>
              <li>
                Initial pilot scope is county-level, using ACLED, CHIRPS, MODIS
                NDVI, OSM, and optional market signals.
              </li>
            </ul>
          </div>
        </div>
      </div>
    </section>
  );
}
