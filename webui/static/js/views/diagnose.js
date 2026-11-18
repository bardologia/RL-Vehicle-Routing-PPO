import { PlaceholderView } from "./placeholder.js";

export class DiagnoseView extends PlaceholderView {
  constructor(app, root) {
    super(app, root, "Diagnose", "The view across evaluated episodes is built in package C. Single episodes can be recorded and replayed in Episode.");
  }
}
