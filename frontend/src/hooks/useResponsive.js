import { useState, useEffect } from 'react';
// Coarse-pointer / touch capability, independent of viewport width. A large
// tablet or a touch laptop can be >=1024px wide (isDesktop) yet have no hover
// and a fat finger for a pointer — control affordances key off THIS, not width,
// so the preview players stay usable wherever there's no mouse.
import { detectTouch } from '../utils/playerControls';

export default function useResponsive() {
  const [state, setState] = useState(() => {
    const w = typeof window !== 'undefined' ? window.innerWidth : 1024;
    return {
      isMobile: w < 768,
      isTablet: w >= 768 && w < 1024,
      isDesktop: w >= 1024,
      isTouch: detectTouch(),
    };
  });

  useEffect(() => {
    const mq = window.matchMedia('(max-width: 767px)');
    const tq = window.matchMedia('(min-width: 768px) and (max-width: 1023px)');
    const pq = window.matchMedia('(pointer: coarse)');
    const update = () => {
      setState({
        isMobile: mq.matches,
        isTablet: tq.matches,
        isDesktop: !mq.matches && !tq.matches,
        isTouch: detectTouch(),
      });
    };
    mq.addEventListener('change', update);
    tq.addEventListener('change', update);
    // A hybrid device can flip pointer type (dock/undock, plug a mouse in).
    pq.addEventListener('change', update);
    return () => {
      mq.removeEventListener('change', update);
      tq.removeEventListener('change', update);
      pq.removeEventListener('change', update);
    };
  }, []);

  return state;
}
