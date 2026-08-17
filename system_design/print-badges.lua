-- PDF filter for System-Design.pdf
-- 1. Convert raw-HTML <img> tags (used for GitHub sizing) into real images.
-- 2. Drop shields.io badge images (web-only decorations).
-- 3. Force every local diagram to full text width (own row, no aspect distortion).
--    Pandoc otherwise emits [width=..,height=\textheight], which distorts
--    aspect ratio; clearing height keeps width-only scaling.

local function process(img)
  local src = img.src or ""
  if src:find("shields%.io") or src:find("img%.shields%.io") then
    return {}
  end
  img.attributes.width = "100%"
  img.attributes.height = "" -- remove forced height -> no stretching
  return img
end

function Image(el)
  return process(el)
end

function RawInline(el)
  if el.format == "html" then
    local src = el.text:match('<img src="([^"]+)"')
    if src then
      return process(pandoc.Image({}, src, ""))
    end
  end
  return el
end

function RawBlock(el)
  if el.format == "html" then
    local src = el.text:match('<img src="([^"]+)"')
    if src then
      return process(pandoc.Image({}, src, ""))
    end
  end
  return el
end
