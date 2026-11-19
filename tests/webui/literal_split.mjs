import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here   = path.dirname(fileURLToPath(import.meta.url));
const module = path.join(here, "..", "..", "webui", "static", "js", "launch", "python_literal.js");
const { PythonLiteral } = await import(pathToFileURL(module).href);

const fields = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
process.stdout.write(JSON.stringify(fields.map((field) => PythonLiteral.split(field))));
