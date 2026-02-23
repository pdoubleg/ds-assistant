/**
 * A2UIRenderer - Dynamic Component Renderer
 *
 * Takes A2UI component specifications from the backend and renders them
 * using the registered catalog of React components.
 */

import React from 'react';
import { motion } from 'framer-motion';
import type { A2UIComponent } from '@/lib/a2ui-catalog';
import { getComponentRenderer, isComponentRegistered } from '@/lib/a2ui-catalog';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

interface A2UIRendererProps {
  /** A2UI component specification from the backend */
  component: A2UIComponent;
  /** Optional class name for the wrapper */
  className?: string;
  /** Callback when a component type is not found in the catalog */
  onMissingComponent?: (type: string) => void;
  /** Whether to show error cards for missing components (default: true) */
  showErrors?: boolean;
}

/**
 * A2UIRenderer Component
 *
 * Recursively renders A2UI component trees by looking up component types
 * in the catalog and passing props to the registered renderers.
 */
export function A2UIRenderer({
  component,
  className,
  onMissingComponent,
  showErrors = true,
}: A2UIRendererProps): React.ReactElement {
  if (!component || !component.type) {
    if (showErrors) {
      return (
        <Card className="border-red-500 bg-red-500/10">
          <CardHeader>
            <CardTitle className="text-sm text-red-700">Invalid Component</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-xs text-red-600">Component specification is missing or invalid.</p>
          </CardContent>
        </Card>
      );
    }
    return <></>;
  }

  const renderer = getComponentRenderer(component.type);

  if (!renderer) {
    if (onMissingComponent) {
      onMissingComponent(component.type);
    }

    if (showErrors) {
      return (
        <Card className="border-yellow-500 bg-yellow-500/10">
          <CardHeader>
            <div className="flex items-center gap-2">
              <CardTitle className="text-sm">Unknown Component</CardTitle>
              <Badge variant="outline">{component.type}</Badge>
            </div>
            <CardDescription className="text-xs">
              Component type not registered in catalog
            </CardDescription>
          </CardHeader>
          <CardContent>
            <details className="text-xs">
              <summary className="cursor-pointer font-medium mb-2">Component Details</summary>
              <pre className="bg-muted p-2 rounded text-xs overflow-x-auto">
                {JSON.stringify(component, null, 2)}
              </pre>
            </details>
          </CardContent>
        </Card>
      );
    }
    return <></>;
  }

  // Render children if present
  const childComponents = component.children?.map((child, index) => (
    <A2UIRenderer
      key={child.id || `child-${index}`}
      component={child}
      onMissingComponent={onMissingComponent}
      showErrors={showErrors}
    />
  ));

  const wrapperClassName = [
    className,
    component.layout?.className,
    component.styling?.className,
  ]
    .filter(Boolean)
    .join(' ');

  const wrapperStyle: React.CSSProperties = {
    ...(component.layout?.width && { width: component.layout.width }),
    ...(component.layout?.height && { height: component.layout.height }),
    ...(component.layout?.position && { position: component.layout.position }),
  };

  const componentProps = {
    ...component.props,
    ...(component.styling?.variant && { variant: component.styling.variant }),
    ...(component.styling?.theme && { theme: component.styling.theme }),
  };

  const renderedComponent = renderer(componentProps, childComponents);

  if (wrapperClassName || Object.keys(wrapperStyle).length > 0) {
    return (
      <div className={wrapperClassName} style={wrapperStyle}>
        {renderedComponent}
      </div>
    );
  }

  return renderedComponent;
}

/**
 * A2UIRendererList Component
 *
 * Renders a list of A2UI components with optional spacing and stagger animation.
 */
interface A2UIRendererListProps {
  components: A2UIComponent[];
  className?: string;
  spacing?: 'none' | 'sm' | 'md' | 'lg';
  onMissingComponent?: (type: string) => void;
  showErrors?: boolean;
}

export function A2UIRendererList({
  components,
  className,
  spacing = 'md',
  onMissingComponent,
  showErrors = true,
}: A2UIRendererListProps): React.ReactElement {
  const spacingClasses = {
    none: '',
    sm: 'space-y-2',
    md: 'space-y-4',
    lg: 'space-y-6',
  };

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.1 },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.4, ease: "easeOut" as const },
    },
  };

  return (
    <motion.div
      className={`${spacingClasses[spacing]} ${className || ''}`}
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      {components.map((component, index) => (
        <motion.div key={component.id || `component-${index}`} variants={itemVariants}>
          <A2UIRenderer
            component={component}
            onMissingComponent={onMissingComponent}
            showErrors={showErrors}
          />
        </motion.div>
      ))}
    </motion.div>
  );
}

export default A2UIRenderer;
