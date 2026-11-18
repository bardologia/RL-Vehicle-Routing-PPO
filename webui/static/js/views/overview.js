import { PlaceholderView } from "./placeholder.js";

export class OverviewView extends PlaceholderView {
  constructor(app, root) {
    super(app, root, "Overview", "Model status, findings and checkpoints are built in package C. Episode and Lab are ready.");
  }
}
