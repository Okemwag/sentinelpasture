"use client";

import React from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import Image from "next/image";
import { HeroHeader } from "@/components/hero8-header";
import { InfiniteSlider } from "@/components/ui/infinite-slider";
import { ProgressiveBlur } from "@/components/ui/progressive-blur";

export default function HeroSection() {
  const heroRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    const handleScroll = () => {
      if (heroRef.current) {
        const scrollY = window.scrollY;
        heroRef.current.style.setProperty('--scroll-y', `${scrollY * 0.5}px`);
      }
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <>
      <HeroHeader />
      <main className="overflow-x-hidden">
        <section ref={heroRef}>
          <div className="pb-24 pt-12 md:pb-32 lg:pb-56 lg:pt-44">
            <div className="relative mx-auto flex max-w-6xl flex-col px-6 lg:block">
              <div className="mx-auto max-w-lg text-center lg:ml-0 lg:w-1/2 lg:text-left">
                <h1 className="mt-8 max-w-2xl text-balance text-5xl font-medium md:text-6xl lg:mt-16 xl:text-7xl">
                  Restoring the state&apos;s ability to govern conditions, not just events.
                </h1>
                <p className="mt-8 max-w-2xl text-pretty text-lg">
                  We are building a governance intelligence system for detecting instability early, understanding why it is emerging, and coordinating proportionate response before crises escalate.
                </p>

                <div className="mt-12 flex flex-col items-center justify-center gap-2 sm:flex-row lg:justify-start">
                  <Button asChild size="lg" className="px-5 text-base">
                    <Link href="#contact">
                      <span className="text-nowrap">Get in Touch</span>
                    </Link>
                  </Button>
                  <Button
                    key={2}
                    asChild
                    size="lg"
                    variant="ghost"
                    className="px-5 text-base"
                  >
                    <Link href="/about">
                      <span className="text-nowrap">Learn More</span>
                    </Link>
                  </Button>
                </div>
              </div>
              <Image
                className="-z-10 order-first ml-auto h-56 w-full object-cover invert sm:h-96 lg:absolute lg:inset-0 lg:-right-20 lg:-top-96 lg:order-last lg:h-max lg:w-2/3 lg:object-contain dark:mix-blend-lighten dark:invert-0 transition-transform duration-300 ease-out"
                style={{
                  transform: 'translateY(var(--scroll-y, 0px))',
                }}
                src="https://res.cloudinary.com/dg4jhba5c/image/upload/v1741605150/abstract-bg_wq4f8w.jpg"
                alt="Abstract Object"
                height="4000"
                width="3000"
                priority
              />
            </div>
          </div>
        </section>
      </main>
    </>
  );
}
