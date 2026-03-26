import { useTheme } from '@/contexts/ThemeContext';

/**
 * LiE 워드마크 — 원본 SVG(LiE_Light_Commercial.svg) 좌표 기반 재구현.
 * 원본 viewBox: 0 0 1400 788, 콘텐츠 영역: x400~988, y208~512
 * 배경 rect 제거 + 테마 반응형 색상 적용.
 */
interface LiELogoProps {
  height?: number;
  showTagline?: boolean;
}

export default function LiELogo({ height = 32, showTagline = false }: LiELogoProps) {
  const { theme } = useTheme();

  const letterColor = theme === 'dark' ? '#dde6f0' : '#0D1B2A';
  const accentColor = '#F5A623';
  const taglineColor = theme === 'dark' ? '#3a6b8a' : '#3A6B8A';

  /* 콘텐츠만 보이도록 viewBox 축소 (원본 좌표 기준) */
  const contentVB = '398 206 594 314'; /* x, y, w, h — 살짝 여백 추가 */
  const aspect = 594 / 314;
  const width = Math.round(height * aspect);

  const taglineVB = '398 206 594 460';
  const taglineHeight = Math.round(height * (460 / 314));

  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox={showTagline ? taglineVB : contentVB}
      width={width}
      height={showTagline ? taglineHeight : height}
      aria-label="LiE — Light Inference Engine"
    >
      {/* ── L ─────────────────────────────────────────── */}
      <rect x="412" y="228" width="76" height="284" fill={letterColor} />
      <rect x="412" y="436" width="196" height="76"  fill={letterColor} />

      {/* ── i ─────────────────────────────────────────── */}
      <circle cx="694" cy="244" r="36"               fill={accentColor} />
      <rect x="668" y="288" width="52"  height="224" fill={accentColor} />

      {/* ── E ─────────────────────────────────────────── */}
      <rect x="784" y="228" width="76"  height="284" fill={letterColor} />
      <rect x="784" y="228" width="204" height="76"  fill={letterColor} />
      <rect x="784" y="354" width="168" height="58"  fill={letterColor} />
      <rect x="784" y="436" width="204" height="76"  fill={letterColor} />

      {/* ── Tagline (optional) ────────────────────────── */}
      {showTagline && (
        <text
          x="695.5" y="630"
          fontFamily="'Montserrat', 'Helvetica Neue', sans-serif"
          fontSize="28"
          fontWeight="300"
          fill={taglineColor}
          textAnchor="middle"
          letterSpacing="9"
        >
          Light Inference Engine
        </text>
      )}
    </svg>
  );
}
