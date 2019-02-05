<?php

declare(strict_types = 1);

namespace drupol\cgcl\Command;

use Gitonomy\Git\Repository;
use Phpml\FeatureExtraction\TfIdfTransformer;
use Phpml\FeatureExtraction\TokenCountVectorizer;
use Phpml\ModelManager;
use Phpml\Tokenization\WordTokenizer;
use Symfony\Component\Console\Command\Command;
use Symfony\Component\Console\Input\InputArgument;
use Symfony\Component\Console\Input\InputInterface;
use Symfony\Component\Console\Output\OutputInterface;

class Predict extends Command
{
    /**
     * {@inheritdoc}
     */
    protected function configure()
    {
        $this->setName('predict')
            ->setDescription('Predict')
            ->setHelp('')
            ->addArgument('repository', InputArgument::REQUIRED, 'Git repository')
            ->addArgument('string', InputArgument::REQUIRED, 'string');
    }

    /**
     * {@inheritdoc}
     */
    protected function execute(InputInterface $input, OutputInterface $output)
    {
        // Create a unique file per repository.
        $filepath = 'cache/' . \sha1($input->getArgument('repository')) . '.phpml';

        if (!\file_exists($filepath)) {
            die('Cache file doesnt exists.');
        }

        $modelManager = new ModelManager();
        $classifier = $modelManager->restoreFromFile($filepath);

        $samples = [$input->getArgument('string')];

        // Should these two objects be created out of the loop ?
        $vectorizer = new TokenCountVectorizer(new WordTokenizer());
        $tfIdfTransformer = new TfIdfTransformer();

        $vectorizer->fit($samples);
        $vectorizer->transform($samples);

        $tfIdfTransformer->fit($samples);
        $tfIdfTransformer->transform($samples);

        $prediction = $classifier->predict($samples);
        $output->writeln(\sprintf('Prediction... '));
        $output->writeln(\print_r($prediction, true));
    }
}
