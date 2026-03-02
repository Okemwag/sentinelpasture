"use client";

import { Button } from "@/components/ui/button";
import { ArrowLeft } from "lucide-react";
import { useRouter } from "next/navigation";

export function BackButton() {
    const router = useRouter();

    return (
        <Button
            onClick={() => router.back()}
            variant="ghost"
            size="sm"
            className="mb-8"
        >
            <ArrowLeft className="mr-2 size-4" />
            Back
        </Button>
    );
}
