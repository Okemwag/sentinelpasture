interface PageHeaderProps {
  title: string;
  children?: React.ReactNode;
}

export default function PageHeader({ title, children }: PageHeaderProps) {
  return (
    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-6">
      <h1 className="text-[24px] sm:text-[28px] font-semibold text-[#111111]">{title}</h1>
      {children && <div className="flex-shrink-0">{children}</div>}
    </div>
  );
}
