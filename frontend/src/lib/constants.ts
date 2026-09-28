import { MachineInputData } from "../types/api";

export const DEFAULT_INPUT: MachineInputData = {
  type: "M",
  air_temperature: 298.1,
  process_temperature: 308.6,
  rotational_speed: 1551.0,
  torque: 42.8,
  tool_wear: 0.0,
};

export interface ExamplePreset {
  name: string;
  description: string;
  data: MachineInputData;
}

export const EXAMPLE_PRESETS: ExamplePreset[] = [
  {
    name: "Typical Operation",
    description: "Standard parameters observed during normal cutting cycles.",
    data: {
      type: "M",
      air_temperature: 298.1,
      process_temperature: 308.6,
      rotational_speed: 1551.0,
      torque: 42.8,
      tool_wear: 25.0,
    },
  },
  {
    name: "Elevated Mechanical Load",
    description: "Operating with high spindle torque and reduced speed.",
    data: {
      type: "L",
      air_temperature: 299.2,
      process_temperature: 309.8,
      rotational_speed: 1320.0,
      torque: 69.5,
      tool_wear: 105.0,
    },
  },
  {
    name: "High Tool Wear",
    description: "Tool approaching end of service life under active machining load.",
    data: {
      type: "L",
      air_temperature: 298.8,
      process_temperature: 309.3,
      rotational_speed: 1410.0,
      torque: 58.0,
      tool_wear: 235.0,
    },
  },
];

export const FIELD_DEFINITIONS = {
  type: {
    label: "Machine Type",
    description: "Product quality tier (L: 60.00%, M: 29.97%, H: 10.03%)",
  },
  air_temperature: {
    label: "Air Temperature",
    unit: "K",
    min: 280.0,
    max: 320.0,
    step: 0.1,
    observedMin: 295.3,
    observedMax: 304.4,
    description: "Ambient shop floor air temperature in Kelvin",
  },
  process_temperature: {
    label: "Process Temperature",
    unit: "K",
    min: 290.0,
    max: 330.0,
    step: 0.1,
    observedMin: 305.7,
    observedMax: 313.8,
    description: "Cutting zone process temperature in Kelvin",
  },
  rotational_speed: {
    label: "Rotational Speed",
    unit: "rpm",
    min: 500,
    max: 3500,
    step: 1,
    observedMin: 1168,
    observedMax: 2886,
    description: "Spindle rotation speed in revolutions per minute",
  },
  torque: {
    label: "Torque",
    unit: "Nm",
    min: 0.0,
    max: 120.0,
    step: 0.1,
    observedMin: 3.8,
    observedMax: 76.6,
    description: "Drive motor torque in Newton-meters",
  },
  tool_wear: {
    label: "Tool Wear",
    unit: "min",
    min: 0,
    max: 400,
    step: 1,
    observedMin: 0,
    observedMax: 253,
    description: "Cumulative cutting insert operating time in minutes",
  },
};
