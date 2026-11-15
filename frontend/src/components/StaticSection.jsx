import fundusUrl from "../../img/fundus.jpg";
import maskUrl from "../../img/mask.jpg";

export default function StaticSection({ markup }) {
  const html = markup
    .replaceAll("/img/fundus.jpg", fundusUrl)
    .replaceAll("/img/mask.jpg", maskUrl);

  return <div dangerouslySetInnerHTML={{ __html: html }} />;
}
