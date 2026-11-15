import markup from "../overview.html?raw";
import StaticSection from "./StaticSection.jsx";

export default function Overview() {
  return <StaticSection markup={markup} />;
}
