interface PageHeaderProps {
  title: string;
  children?: React.ReactNode;
}

export default function PageHeader({ title, children }: PageHeaderProps) {
  return (
    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-5">
      <div>
        <h1 className="text-[18px] sm:text-[20px] font-semibold tracking-tight text-[var(--intel-text-primary)]">
          {title}
        </h1>
        <p className="mt-1 text-[12px] text-[var(--intel-text-secondary)]">
          Governance-grade intelligence view for this dimension.
        </p>
      </div>
      {children && <div className="flex-shrink-0">{children}</div>}
    </div>
  );
}
