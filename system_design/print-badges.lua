-- PDF filter for System-Design.pdf
-- 1. Convert raw-HTML <img> tags (used for GitHub sizing) into real images.
-- 2. Drop shields.io badge images (web-only decorations).
-- 3. Cap local diagram widths so tall figures (CNN) and wide figures fit a page.

local function process(img)
  local src = img.src or ""
  if src:find("shields%.io") or src:find("img%.shields%.io") then
    return {}
  end
  local name = src:match("([^/]+)$") or ""
  if name:find("cnn%-architecture") then
    img.attributes.width = "56%"
  elseif name:find("%.png$") or name:find("%.svg$") then
    img.attributes.width = "82%"
  end
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
