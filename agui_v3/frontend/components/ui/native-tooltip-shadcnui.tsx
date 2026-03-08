"use client";

import * as TooltipPrimitive from "@radix-ui/react-tooltip";
import { AnimatePresence, motion, type TargetAndTransition, type Transition } from "framer-motion";
import * as React from "react";

import { cn } from "@/lib/utils";
import { ShineBorder } from "@/components/ui/shine-border";

/** Shared shine colours — matches the agent action buttons. */
const SHINE_COLORS: string[] = ["#A07CFE", "#FE8FB5", "#FFBE7B"];

const NativeTooltipProvider = ({
  delayDuration = 100,
  ...props
}: React.ComponentProps<typeof TooltipPrimitive.Provider>) => (
  <TooltipPrimitive.Provider delayDuration={delayDuration} {...props} />
);

const NativeTooltipRoot = TooltipPrimitive.Root;

const NativeTooltipTrigger = TooltipPrimitive.Trigger;

const NativeTooltipContent = React.forwardRef<
  React.ComponentRef<typeof TooltipPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof TooltipPrimitive.Content> & {
    animation?: "blur" | "scale";
    /** Show the animated shine border (matches agent action buttons). */
    shine?: boolean;
  }
>(
  (
    { className, sideOffset = 8, children, animation = "blur", shine = false, ...props },
    ref
  ) => {
    const animations: Record<
      "blur" | "scale",
      {
        initial: TargetAndTransition;
        animate: TargetAndTransition;
        exit: TargetAndTransition;
        transition: Transition;
      }
    > = {
      blur: {
        initial: { opacity: 0, scale: 0.9, filter: "blur(4px)" },
        animate: { opacity: 1, scale: 1, filter: "blur(0px)" },
        exit: { opacity: 0, scale: 0.9, filter: "blur(4px)" },
        transition: { type: "spring", duration: 0.3, bounce: 0 },
      },
      scale: {
        initial: { opacity: 0, scale: 0.5, y: 10 },
        animate: { opacity: 1, scale: 1, y: 0 },
        exit: { opacity: 0, scale: 0.5, y: 10 },
        transition: { type: "spring", duration: 0.3, bounce: 0.4 },
      },
    };

    const selectedAnimation = animations[animation];

    return (
      <TooltipPrimitive.Portal>
        {/*
         * forceMount keeps the portal in the DOM so AnimatePresence can drive
         * the exit animation before Radix unmounts. We hide it ourselves via
         * the motion variants; Radix's data-state drives the open/closed key.
         */}
        <TooltipPrimitive.Content
          ref={ref}
          forceMount
          sideOffset={sideOffset}
          className={cn("z-50 overflow-visible bg-transparent", className)}
          {...props}
        >
          {/* data-state="delayed-open"|"instant-open"|"closed" */}
          <RadixPresenceAdapter selectedAnimation={selectedAnimation} shine={shine}>
            {children}
          </RadixPresenceAdapter>
        </TooltipPrimitive.Content>
      </TooltipPrimitive.Portal>
    );
  }
);
NativeTooltipContent.displayName = TooltipPrimitive.Content.displayName;

/** Reads Radix's data-state from its parent and drives AnimatePresence. */
const RadixPresenceAdapter = ({
  children,
  selectedAnimation,
  shine,
}: {
  children: React.ReactNode;
  shine?: boolean;
  selectedAnimation: {
    initial: TargetAndTransition;
    animate: TargetAndTransition;
    exit: TargetAndTransition;
    transition: Transition;
  };
}) => {
  // Radix sets data-state on the Content element; we read it via a ref on a
  // wrapper span that lives inside the Content, then walk up one level.
  const wrapperRef = React.useRef<HTMLSpanElement>(null);
  const [isOpen, setIsOpen] = React.useState(false);

  React.useEffect(() => {
    // The direct parent is TooltipPrimitive.Content which Radix marks with
    // data-state="delayed-open" | "instant-open" | "closed"
    const parent = wrapperRef.current?.parentElement;
    if (!parent) return;

    const update = () => {
      const state = parent.getAttribute("data-state");
      setIsOpen(state === "delayed-open" || state === "instant-open");
    };

    update(); // sync on mount
    const observer = new MutationObserver(update);
    observer.observe(parent, { attributes: true, attributeFilter: ["data-state"] });
    return () => observer.disconnect();
  }, []);

  return (
    <>
      {/* invisible anchor so we can walk up to the Radix Content element */}
      <span ref={wrapperRef} style={{ display: "none" }} aria-hidden />
      <AnimatePresence>
        {isOpen && (
          <motion.div
            key="tooltip-content"
            initial={selectedAnimation.initial}
            animate={selectedAnimation.animate}
            exit={selectedAnimation.exit}
            transition={selectedAnimation.transition}
            className="relative overflow-hidden rounded-md border border-white/10 bg-black/80 dark:bg-white/90 backdrop-blur-md px-3 py-1.5 text-xs font-medium text-white dark:text-black shadow-lg"
          >
            {/* optional shine border — thinner and faster than the button variant */}
            {shine && (
              <ShineBorder
                borderWidth={0.75}
                duration={10}
                shineColor={SHINE_COLORS}
              />
            )}
            {children}
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
};

const NativeTooltip = ({
  content,
  children,
  animation,
  side,
  sideOffset,
  align,
  shine,
  contentProps,
  ...props
}: React.ComponentProps<typeof TooltipPrimitive.Root> & {
  content?: React.ReactNode;
  animation?: "blur" | "scale";
  /** Placement side of the tooltip content. */
  side?: React.ComponentPropsWithoutRef<typeof NativeTooltipContent>["side"];
  /** Offset in px from the trigger. */
  sideOffset?: number;
  /** Alignment relative to trigger. */
  align?: React.ComponentPropsWithoutRef<typeof NativeTooltipContent>["align"];
  /** Show the animated shine border on the tooltip bubble. */
  shine?: boolean;
  /** Any additional props forwarded to NativeTooltipContent. */
  contentProps?: Omit<
    React.ComponentPropsWithoutRef<typeof NativeTooltipContent>,
    "side" | "sideOffset" | "align" | "animation" | "shine"
  >;
}) => {
  if (content) {
    return (
      <NativeTooltipRoot {...props}>
        <NativeTooltipTrigger asChild>{children}</NativeTooltipTrigger>
        <NativeTooltipContent
          animation={animation}
          side={side}
          sideOffset={sideOffset}
          align={align}
          shine={shine}
          {...contentProps}
        >
          {content}
        </NativeTooltipContent>
      </NativeTooltipRoot>
    );
  }

  return <NativeTooltipRoot {...props}>{children}</NativeTooltipRoot>;
};

export {
  NativeTooltip,
  NativeTooltipContent,
  NativeTooltipProvider,
  NativeTooltipTrigger,
};
