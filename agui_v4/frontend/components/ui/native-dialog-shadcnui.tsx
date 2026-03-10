"use client";

import {
  DialogDescription as BaseDialogDescription,
  DialogTitle as BaseDialogTitle,
  Dialog,
  DialogClose,
  DialogFooter,
  DialogHeader,
  DialogPortal,
  DialogTrigger,
} from "@/components/ui/dialog";
import { cn } from "@/lib/utils";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import { motion } from "framer-motion";
import { X } from "lucide-react";
import * as React from "react";

const NativeDialog = Dialog;

const NativeDialogTrigger = DialogTrigger;

const NativeDialogPortal = DialogPortal;

const NativeDialogClose = DialogClose;

const NativeDialogOverlay = React.forwardRef<
  React.ComponentRef<typeof DialogPrimitive.Overlay>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Overlay>
>(({ className, ...props }, ref) => (
  <DialogPrimitive.Overlay ref={ref} asChild {...props}>
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className={cn(
        "fixed inset-0 z-50 bg-black/50",
        className
      )}
    />
  </DialogPrimitive.Overlay>
));
NativeDialogOverlay.displayName = DialogPrimitive.Overlay.displayName;

const NativeDialogContent = React.forwardRef<
  React.ComponentRef<typeof DialogPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Content>
>(({ className, children, ...props }, ref) => (
  <NativeDialogPortal>
    <NativeDialogOverlay />
    <DialogPrimitive.Content ref={ref} asChild {...props}>
      <motion.div
        initial={{ opacity: 0, scale: 0.97 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.97 }}
        transition={{ type: "spring", duration: 0.5, bounce: 0 }}
        className={cn(
          "fixed left-[50%] top-[50%] z-50 grid w-full max-w-lg translate-x-[-50%] translate-y-[-50%] gap-4 border border-zinc-200 bg-zinc-50 p-6 shadow-xl dark:border-zinc-800 dark:bg-zinc-900 sm:rounded-xl",
          className
        )}
      >
        {children}
        <DialogClose className="absolute right-4 top-4 rounded-md p-1.5 text-zinc-500 opacity-80 ring-offset-background transition-all hover:bg-zinc-100 hover:text-zinc-900 hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:pointer-events-none dark:text-zinc-400 dark:hover:bg-zinc-800 dark:hover:text-zinc-50">
          <X className="h-4 w-4" />
          <span className="sr-only">Close</span>
        </DialogClose>
      </motion.div>
    </DialogPrimitive.Content>
  </NativeDialogPortal>
));
NativeDialogContent.displayName = DialogPrimitive.Content.displayName;

const NativeDialogHeader = DialogHeader as typeof DialogHeader & { displayName?: string };
NativeDialogHeader.displayName = "NativeDialogHeader";

const NativeDialogFooter = DialogFooter as typeof DialogFooter & { displayName?: string };
NativeDialogFooter.displayName = "NativeDialogFooter";

const NativeDialogTitle = BaseDialogTitle as typeof BaseDialogTitle & { displayName?: string };
NativeDialogTitle.displayName = "NativeDialogTitle";

const NativeDialogDescription = BaseDialogDescription as typeof BaseDialogDescription & { displayName?: string };
NativeDialogDescription.displayName = "NativeDialogDescription";

export {
  NativeDialog,
  NativeDialogClose,
  NativeDialogContent,
  NativeDialogDescription,
  NativeDialogFooter,
  NativeDialogHeader,
  NativeDialogOverlay,
  NativeDialogPortal,
  NativeDialogTitle,
  NativeDialogTrigger,
};
