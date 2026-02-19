interface PageHeaderProps {
  title: string;
  children?: React.ReactNode;
}

export default function PageHeader({ title, children }: PageHeaderProps) {
  return (
    <div className="flex items-center justify-between mb-6">
      <h1 className="text-[28px] font-semibold text-[#111111]">{title}</h1>
      {children}
    </div>
  );
}
