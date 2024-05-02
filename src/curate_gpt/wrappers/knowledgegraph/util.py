import logging
import re
from urllib.parse import urlparse

import click
import requests

from curate_gpt.wrappers.knowledgegraph import VALID_IDS, VALID_PHENO

logger = logging.getLogger(__name__)


def validate_list_elements(lst, prefix, valid_set):
    if not all(isinstance(e, str) and e.startswith(prefix) and re.sub(r"\d", "", e) in valid_set for e in lst):
        raise ValueError(f"List validation failed for prefix {prefix} and valid set {valid_set}")
    return lst


def validate_gene_list(gene_list, gene_prefix):
    if not isinstance(gene_list, list) or not gene_list:
        raise ValueError("Expected a non-empty list of strings")
    return validate_list_elements(gene_list, gene_prefix, VALID_IDS)


def validate_prefix(prefix, valid_set):
    if prefix not in valid_set:
        raise ValueError(f"Prefix should be one of {valid_set}, got {prefix}")
    return prefix


def validate_gene_file(file):
    if not file.lower().endswith('.txt'):
        raise ValueError("Only .txt files are allowed for the gene list")
    return file


def validate_prefixis(gene_prefix, phenotype_prefix):
    validate_prefix(gene_prefix, VALID_IDS)
    validate_prefix(phenotype_prefix, VALID_PHENO)
    if gene_prefix == "HGNC:" and phenotype_prefix != "HP:" or gene_prefix == "MGI:" and phenotype_prefix != "MP:":
        raise ValueError(f"Incompatible gene and phenotype prefixes: {gene_prefix} and {phenotype_prefix}")
    return gene_prefix, phenotype_prefix


def validate_monarch_url(url):
    try:
        result = urlparse(url)
        if all([result.scheme, result.netloc, result.path]) and result.scheme in ['http', 'https']:
            if result.netloc != 'data.monarchinitiative.org':
                raise ValueError("URL domain is not valid for Monarch Initiative data.")
            if not re.match(r'^/monarch-kg/\d{4}-\d{2}-\d{2}/.+\.tar\.gz$', result.path):
                raise ValueError("URL path is malformed or points to an incorrect file format.")
    except Exception as e:
        error_message = handle_exception(e)
        raise ValueError(error_message)
    return url


def parse_gene_list(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        if len(lines) == 1:
            genes = [gene.strip() for gene in re.split('[ ,]+', lines[0].strip())]
        else:
            genes = [line.strip() for line in lines]
        return genes
    except Exception as e:
        error_message = handle_exception(e)
        raise RuntimeError(error_message)


def validate_upsert_option(ctx, param, value):
    upsert_mode = ctx.params.get('upsert')
    if upsert_mode in ['gene_by_gene', 'setwise'] and not value:
        raise click.BadParameter('gene_list_file is required for gene_by_gene and setwise modes.')
    return value


def handle_exception(e):
    if isinstance(e, FileNotFoundError):
        return "File not found error: Check the file path."
    elif isinstance(e, ValueError):
        return f"Value error: {e}"
    elif isinstance(e, requests.RequestException):
        return f"Network error when accessing URL: {e}"
    elif isinstance(e, IOError):
        return "Input/output error: Unable to read the file."
    else:
        return "An unexpected error occurred."
