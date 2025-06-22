"use client"

import * as React from "react"
import * as TabsPrimitive from "@radix-ui/react-tabs"

import { cn } from "@/lib/utils"

function Tabs({
  className,
  ...props
}: React.ComponentProps<typeof TabsPrimitive.Root>) {
  return (
    <TabsPrimitive.Root
      data-slot="tabs"
      className={cn("flex flex-col gap-2", className)}
      {...props}
    />
  )
}

function TabsList({
  className,
  ...props
}: React.ComponentProps<typeof TabsPrimitive.List>) {
  const [activeTabInfo, setActiveTabInfo] = React.useState<{
    width: number;
    left: number;
  } | null>(null);
  const listRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    const updateActiveTabPosition = () => {
      if (!listRef.current) return;

      const activeTab = listRef.current.querySelector('[data-state="active"]') as HTMLElement;
      if (activeTab) {
        const listRect = listRef.current.getBoundingClientRect();
        const activeTabRect = activeTab.getBoundingClientRect();
        
        setActiveTabInfo({
          width: activeTabRect.width,
          left: activeTabRect.left - listRect.left,
        });
      }
    };

    // Update position on mount and when tabs change
    updateActiveTabPosition();
    
    // Use ResizeObserver to handle dynamic changes
    const resizeObserver = new ResizeObserver(updateActiveTabPosition);
    if (listRef.current) {
      resizeObserver.observe(listRef.current);
    }

    // Also listen for mutations in case tab content changes
    const mutationObserver = new MutationObserver(updateActiveTabPosition);
    if (listRef.current) {
      mutationObserver.observe(listRef.current, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ['data-state']
      });
    }

    return () => {
      resizeObserver.disconnect();
      mutationObserver.disconnect();
    };
  }, []);

  return (
    <TabsPrimitive.List
      ref={listRef}
      data-slot="tabs-list"
      className={cn(
        "bg-muted text-muted-foreground relative inline-flex h-9 w-fit items-center justify-center rounded-lg p-[3px]",
        className
      )}
      {...props}
    >
      {/* Sliding indicator */}
      {activeTabInfo && (
        <div
          className="absolute top-[3px] bottom-[3px] bg-background rounded-md shadow-sm transition-all duration-300 ease-out z-0"
          style={{
            width: activeTabInfo.width,
            transform: `translateX(${activeTabInfo.left}px)`,
          }}
        />
      )}
      {props.children}
    </TabsPrimitive.List>
  )
}

function TabsTrigger({
  className,
  ...props
}: React.ComponentProps<typeof TabsPrimitive.Trigger>) {
  return (
    <TabsPrimitive.Trigger
      data-slot="tabs-trigger"
      className={cn(
        "focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:outline-ring text-foreground dark:text-muted-foreground inline-flex h-[calc(100%-1px)] flex-1 items-center justify-center gap-1.5 rounded-md border border-transparent px-2 py-1 text-sm font-medium whitespace-nowrap transition-[color,box-shadow] focus-visible:ring-[3px] focus-visible:outline-1 disabled:pointer-events-none disabled:opacity-50 relative z-10 data-[state=active]:text-foreground [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4",
        className
      )}
      {...props}
    />
  )
}

function TabsContent({
  className,
  ...props
}: React.ComponentProps<typeof TabsPrimitive.Content>) {
  return (
    <TabsPrimitive.Content
      data-slot="tabs-content"
      className={cn("flex-1 outline-none", className)}
      {...props}
    />
  )
}

export { Tabs, TabsList, TabsTrigger, TabsContent }
