"use client";
import Link from "next/link";
import { Logo } from "./logo";
import { Menu, X, ChevronDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import React from "react";
import { ModeToggle } from "./mode-toggle";

const menuItems = [
  { name: "About", href: "/about" },
  { name: "Team", href: "/team" },
  { name: "Careers", href: "/careers" },
  { name: "Newsroom", href: "/newsroom" },
];

const platformLinks = [
  { name: "Intelligence Dashboard", href: "/platform/dashboard", description: "Real-time governance monitoring" },
  { name: "Early Warning System", href: "/platform/early-warning", description: "Detect instability signals" },
  { name: "Analysis Engine", href: "/platform/analysis", description: "Deep causal analysis" },
  { name: "Response Coordination", href: "/platform/response", description: "Coordinate interventions" },
  { name: "Data Integration", href: "/platform/integration", description: "Connect your data sources" },
];

const solutionsLinks = [
  { name: "Public Safety", href: "/solutions/public-safety", description: "Crime and safety monitoring" },
  { name: "Economic Stability", href: "/solutions/economic", description: "Economic indicators tracking" },
  { name: "Social Cohesion", href: "/solutions/social", description: "Community health metrics" },
  { name: "Infrastructure", href: "/solutions/infrastructure", description: "Critical systems monitoring" },
  { name: "Environmental", href: "/solutions/environmental", description: "Environmental risk assessment" },
];

const resourcesLinks = [
  { name: "Documentation", href: "/resources/docs", description: "Technical guides and API docs" },
  { name: "Case Studies", href: "/resources/case-studies", description: "Real-world applications" },
  { name: "Research Papers", href: "/resources/research", description: "Academic publications" },
  { name: "Blog", href: "/resources/blog", description: "Insights and updates" },
  { name: "Support Center", href: "/resources/support", description: "Get help and training" },
];

interface DropdownMenuProps {
  title: string;
  links: Array<{ name: string; href: string; description: string }>;
}

const DropdownMenu: React.FC<DropdownMenuProps> = ({ title, links }) => {
  const [isOpen, setIsOpen] = React.useState(false);

  return (
    <div
      className="relative"
      onMouseEnter={() => setIsOpen(true)}
      onMouseLeave={() => setIsOpen(false)}
    >
      <button className="text-muted-foreground hover:text-accent-foreground flex items-center gap-1 pb-1 relative group">
        <span>{title}</span>
        <ChevronDown className={`h-4 w-4 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`} />
        <span className="absolute bottom-0 left-0 w-0 h-0.5 bg-white transition-all duration-500 ease-out group-hover:w-full"></span>
      </button>

      {isOpen && (
        <div className="absolute left-0 top-full pt-2 z-50">
          <div className="bg-background rounded-lg shadow-lg p-2 min-w-[280px]">
            {links.map((link, index) => (
              <Link
                key={index}
                href={link.href}
                className="block px-4 py-3 rounded-md hover:bg-accent transition-colors duration-150"
              >
                <div className="font-medium text-sm">{link.name}</div>
                <div className="text-xs text-muted-foreground mt-0.5">{link.description}</div>
              </Link>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

const MobileDropdown: React.FC<DropdownMenuProps> = ({ title, links }) => {
  const [isOpen, setIsOpen] = React.useState(false);

  return (
    <div className="space-y-2">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="text-muted-foreground hover:text-accent-foreground flex items-center justify-between w-full duration-150"
      >
        <span className="font-medium">{title}</span>
        <ChevronDown className={`h-4 w-4 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {isOpen && (
        <div className="pl-4 space-y-3 border-l-2 border-accent/30">
          {links.map((link, index) => (
            <Link
              key={index}
              href={link.href}
              className="block"
            >
              <div className="text-sm font-medium">{link.name}</div>
              <div className="text-xs text-muted-foreground mt-0.5">{link.description}</div>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
};

export const HeroHeader = () => {
  const [menuState, setMenuState] = React.useState(false);
  const [indicatorStyle, setIndicatorStyle] = React.useState({ left: 0, width: 0, opacity: 0 });
  const navRef = React.useRef<HTMLUListElement>(null);

  const handleMouseEnter = (event: React.MouseEvent<HTMLElement>) => {
    const target = event.currentTarget;
    const navList = navRef.current;

    if (navList) {
      const navRect = navList.getBoundingClientRect();
      const targetRect = target.getBoundingClientRect();

      setIndicatorStyle({
        left: targetRect.left - navRect.left,
        width: targetRect.width,
        opacity: 1,
      });
    }
  };

  const handleMouseLeave = () => {
    setIndicatorStyle(prev => ({ ...prev, opacity: 0 }));
  };

  return (
    <header>
      <nav
        data-state={menuState && "active"}
        className="bg-background/50 fixed z-20 w-full border-b backdrop-blur-3xl"
      >
        <div className="mx-auto max-w-6xl px-6 transition-all duration-300">
          <div className="relative flex flex-wrap items-center justify-between gap-6 py-3 lg:gap-0 lg:py-4">
            <div className="flex w-full items-center justify-between gap-12 lg:w-auto">
              <Link
                href="/"
                aria-label="home"
                className="flex items-center space-x-2"
              >
                <Logo />
              </Link>

              <button
                onClick={() => setMenuState(!menuState)}
                aria-label={menuState == true ? "Close Menu" : "Open Menu"}
                className="relative z-20 -m-2.5 -mr-4 block cursor-pointer p-2.5 lg:hidden"
              >
                <Menu className="in-data-[state=active]:rotate-180 in-data-[state=active]:scale-0 in-data-[state=active]:opacity-0 m-auto size-6 duration-200" />
                <X className="in-data-[state=active]:rotate-0 in-data-[state=active]:scale-100 in-data-[state=active]:opacity-100 absolute inset-0 m-auto size-6 -rotate-180 scale-0 opacity-0 duration-200" />
              </button>

              <div className="hidden lg:block">
                <ul ref={navRef} className="flex gap-8 text-sm items-center relative">
                  <li>
                    <div onMouseEnter={handleMouseEnter} onMouseLeave={handleMouseLeave}>
                      <DropdownMenu title="Platform" links={platformLinks} />
                    </div>
                  </li>
                  <li>
                    <div onMouseEnter={handleMouseEnter} onMouseLeave={handleMouseLeave}>
                      <DropdownMenu title="Solutions" links={solutionsLinks} />
                    </div>
                  </li>
                  <li>
                    <div onMouseEnter={handleMouseEnter} onMouseLeave={handleMouseLeave}>
                      <DropdownMenu title="Resources" links={resourcesLinks} />
                    </div>
                  </li>
                  {menuItems.map((item, index) => (
                    <li key={index}>
                      <Link
                        href={item.href}
                        onMouseEnter={handleMouseEnter}
                        onMouseLeave={handleMouseLeave}
                        className="text-muted-foreground hover:text-accent-foreground block pb-1"
                      >
                        <span>{item.name}</span>
                      </Link>
                    </li>
                  ))}
                  {/* Sliding indicator */}
                  <span
                    className="absolute bottom-0 h-0.5 bg-white transition-all duration-300 ease-out"
                    style={{
                      left: `${indicatorStyle.left}px`,
                      width: `${indicatorStyle.width}px`,
                      opacity: indicatorStyle.opacity,
                    }}
                  />
                </ul>
              </div>
            </div>

            <div className="bg-background in-data-[state=active]:block lg:in-data-[state=active]:flex mb-6 hidden w-full flex-wrap items-center justify-end space-y-8 rounded-3xl border p-6 shadow-2xl shadow-zinc-300/20 md:flex-nowrap lg:m-0 lg:flex lg:w-fit lg:gap-6 lg:space-y-0 lg:border-transparent lg:bg-transparent lg:p-0 lg:shadow-none dark:shadow-none dark:lg:bg-transparent">
              <div className="lg:hidden w-full">
                <div className="space-y-6 text-base">
                  <MobileDropdown title="Platform" links={platformLinks} />
                  <MobileDropdown title="Solutions" links={solutionsLinks} />
                  <MobileDropdown title="Resources" links={resourcesLinks} />

                  <div className="border-t pt-6 space-y-4">
                    {menuItems.map((item, index) => (
                      <Link
                        key={index}
                        href={item.href}
                        className="text-muted-foreground hover:text-accent-foreground block duration-150"
                      >
                        <span>{item.name}</span>
                      </Link>
                    ))}
                  </div>
                </div>
              </div>
              <div className="flex w-full flex-col space-y-3 sm:flex-row sm:gap-3 sm:space-y-0 md:w-fit">
                <ModeToggle />
              </div>
            </div>
          </div>
        </div>
      </nav>
    </header>
  );
};
