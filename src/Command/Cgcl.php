<?php

declare(strict_types = 1);

namespace drupol\cgcl\Command;

use drupol\cgcl\Git\CgclCommitParser;
use Gitonomy\Git\Repository;
use Phpml\Classification\MLPClassifier;
use Phpml\Dataset\ArrayDataset;
use Phpml\FeatureExtraction\TfIdfTransformer;
use Phpml\FeatureExtraction\TokenCountVectorizer;
use Phpml\ModelManager;
use Phpml\Tokenization\WordTokenizer;
use Symfony\Component\Console\Command\Command;
use Symfony\Component\Console\Input\InputArgument;
use Symfony\Component\Console\Input\InputInterface;
use Symfony\Component\Console\Output\OutputInterface;

class Cgcl extends Command
{
    /**
     * {@inheritdoc}
     */
    protected function configure()
    {
        $this->setName('learn')
            ->setDescription('Learn')
            ->setHelp('')
            ->addArgument('repository', InputArgument::REQUIRED, 'Git repository');
    }

    /**
     * {@inheritdoc}
     */
    protected function execute(InputInterface $input, OutputInterface $output)
    {
        $modelManager = new ModelManager();
        $cgclcommit = new CgclCommitParser();

        // Connect to the Git repository.
        $repository = new Repository($input->getArgument('repository'));
        $commitCount = $repository->getLog()->count();

        // Create a unique file per repository.
        $filepath = 'cache/' . sha1($input->getArgument('repository')) . '.phpml';

        $classes = [];
        foreach ($repository->getLog() as $commit) {
            $message = $commit->getMessage();
            $classes[\sha1($message)] = $message;
        }

        /** @var \Gitonomy\Git\Commit $commit */
        foreach ($repository->getLog() as $key => $commit) {
            $output->writeln('Processing commit ' . $key . ' / ' . $commitCount . '.');

            if (file_exists($filepath)) {
                $output->writeln(' Restoring data from file...' . $filepath . '...');
                $classifier = $modelManager->restoreFromFile($filepath);
            } else {
                $output->writeln(' Creating new classifier...');
                $classifier = new MLPClassifier(1, [2], $classes);
            }

            // Should these two objects be created out of the loop ?
            $vectorizer = new TokenCountVectorizer(new WordTokenizer());
            $tfIdfTransformer = new TfIdfTransformer();

            // Parse the diff and extract only lines starting with + or -
            $diff = $cgclcommit->parseDiff($commit->getDiff()->getRawDiff());
            $samples = \array_merge($diff['-'], $diff['+']);

            $output->writeln(' Vectorizing commit #' . $key . '.');
            $vectorizer->fit($samples);
            $vectorizer->transform($samples);

            $tfIdfTransformer->fit($samples);
            $tfIdfTransformer->transform($samples);

            $targets = \array_fill(0, \count($samples), $commit->getMessage());

            $dataset = new ArrayDataset($samples, $targets);

            $output->writeln(' Partial training commit #' . $key);
            $classifier->partialTrain($dataset->getSamples(), $dataset->getTargets());

            $output->writeln(' Saving partial data to file... ' . $filepath);
            $modelManager->saveToFile($classifier, $filepath);
        }
    }
}
