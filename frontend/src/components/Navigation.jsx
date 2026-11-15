import markup from "../navigation.html?raw";
import StaticSection from "./StaticSection.jsx";

export default function Navigation() {
  return <StaticSection markup={markup} />;
}
