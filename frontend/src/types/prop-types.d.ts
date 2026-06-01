declare module 'prop-types' {
  export interface Validator<T = any> {
    (props: any, propName: string, componentName: string, ...rest: any[]): Error | null;
    isRequired: Validator<T>;
  }

  export const any: Validator;
  export const array: Validator<any[]>;
  export const bool: Validator<boolean>;
  export const func: Validator<Function>;
  export const number: Validator<number>;
  export const object: Validator<object>;
  export const string: Validator<string>;
  export const symbol: Validator<symbol>;
  export const node: Validator<any>;
  export const element: Validator<any>;
  export const instanceOf: <T>(type: new (...args: any[]) => T) => Validator<T>;
  export const oneOf: <T>(types: ReadonlyArray<T>) => Validator<T>;
  export const oneOfType: <T>(types: Validator[]) => Validator<T>;
  export const arrayOf: <T>(type: Validator<T>) => Validator<T[]>;
  export const objectOf: <T>(type: Validator<T>) => Validator<{ [key: string]: T }>;
  export const shape: <T extends object>(type: T) => Validator<T>;
  export const exact: <T extends object>(type: T) => Validator<T>;

  export function checkPropTypes(
    propTypes: any,
    props: any,
    location: string,
    componentName: string,
    getStack?: () => any
  ): void;

  export function resetWarningCache(): void;
}
