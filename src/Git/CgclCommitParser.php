<?php

declare(strict_types = 1);

namespace drupol\cgcl\Git;

class CgclCommitParser
{
    /**
     * @param string $diff
     *
     * @return array
     */
    public function parseDiff($diff): array
    {
        $lines = ['-' => [], '+' => []];

        foreach (\explode("\n", $diff) as $line) {
            \preg_match('/^(?P<type>[+ -])?(?P<line>.*)/', $line, $match);

            if (0 === \strpos($line, '---', 0) || 0 === \strpos($line, '+++', 0)) {
                continue;
            }

            if ('-' === $match['type']) {
                $lines['-'][] = $line;
            }

            if ('+' === $match['type']) {
                $lines['+'][] = $line;
            }
        }

        return $lines;
    }
}
