'use client';

import React from 'react';

interface BodyWrapperProps {
  children: React.ReactNode;
  className: string;
}

const BodyWrapper = ({ children, className }: BodyWrapperProps) => {
  return (
    <body className={className}>
      {children}
    </body>
  );
};

export default BodyWrapper; 