#!/bin/bash

# sed_change.sh - Optimized script for safely replacing strings in source files
# Usage: ./sed_change.sh <old_string> <new_string> [directory]

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 <old_string> <new_string> [directory]

Description:
    Safely replace strings in source files (.c, .h, .jdf, .txt) with backup creation.

Arguments:
    old_string    The string to search for and replace
    new_string    The string to replace with
    directory     Optional: target directory (default: current directory)

Examples:
    $0 'old_function_name' 'new_function_name'
    $0 'old_variable' 'new_variable' src/
    $0 'deprecated_api' 'new_api' .

Options:
    -h, --help    Show this help message
    -r, --recursive  Process subdirectories recursively
    -b, --backup     Create backup before making changes (disabled by default)
    -n, --dry-run    Show what would be changed without making changes

Safety Features:
    - Creates timestamped backups
    - Validates input parameters
    - Checks file existence
    - Confirms before processing
    - Shows summary of changes
EOF
}

# Function to validate arguments
validate_args() {
    if [ $# -lt 2 ]; then
        print_error "Insufficient arguments provided"
        show_usage
        exit 1
    fi

    if [ $# -gt 3 ]; then
        print_error "Too many arguments provided"
        show_usage
        exit 1
    fi

    # Check if old_string is not empty
    if [ -z "$1" ]; then
        print_error "Old string cannot be empty"
        exit 1
    fi

    # Check if new_string is not empty
    if [ -z "$2" ]; then
        print_error "New string cannot be empty"
        exit 1
    fi

    # Check if old_string and new_string are different
    if [ "$1" = "$2" ]; then
        print_warning "Old string and new string are identical. No changes needed."
        exit 0
    fi
}

# Function to check if directory exists and contains source files
check_directory() {
    local dir="$1"
    
    if [ ! -d "$dir" ]; then
        print_error "Directory '$dir' does not exist"
        exit 1
    fi

    # Check if directory contains source files
    local source_files=$(find "$dir" -maxdepth 1 \( -name "*.c" -o -name "*.h" -o -name "*.jdf" -o -name "*.txt" \) 2>/dev/null | head -1)
    if [ -z "$source_files" ]; then
        print_warning "No source files (.c, .h, .jdf, .txt) found in directory '$dir'"
        echo "Are you sure you want to continue? (y/N)"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            print_info "Operation aborted by user"
            exit 0
        fi
    fi
}

# Function to create backup
create_backup() {
    local dir="$1"
    local backup_dir="backup_$(date +%Y%m%d_%H%M%S)_$(basename "$dir")"
    
    print_info "Creating backup in: $backup_dir"
    mkdir -p "$backup_dir"
    
    # Copy all source files to backup
    find "$dir" \( -name "*.c" -o -name "*.h" -o -name "*.jdf" -o -name "*.txt" \) | while read -r file; do
        if [ -f "$file" ]; then
            local rel_path="${file#$dir}"
            local backup_file="$backup_dir$rel_path"
            mkdir -p "$(dirname "$backup_file")"
            cp "$file" "$backup_file"
        fi
    done
    
    echo "$backup_dir"
}

# Function to perform string replacement
perform_replacement() {
    local dir="$1"
    local old_string="$2"
    local new_string="$3"
    local dry_run="$4"
    local recursive="$5"
    
    local find_cmd="find \"$dir\""
    if [ "$recursive" = "false" ]; then
        find_cmd="$find_cmd -maxdepth 1"
    fi
    find_cmd="$find_cmd \\( -name \"*.c\" -o -name \"*.h\" -o -name \"*.jdf\" -o -name \"*.txt\" \\)"
    
    local total_files=0
    local changed_files=0
    
    while IFS= read -r -d '' file; do
        if [ -f "$file" ]; then
            ((total_files++))
            
            # Check if file contains the old string
            if grep -q "$old_string" "$file" 2>/dev/null; then
                ((changed_files++))
                
                if [ "$dry_run" = "true" ]; then
                    print_info "Would change: $file"
                    grep -n "$old_string" "$file" | head -3 | while IFS=: read -r line_num content; do
                        echo "  Line $line_num: $content"
                    done
                else
                    print_info "Processing: $file"
                    # Create backup of individual file
                    cp "$file" "${file}.bak"
                    # Perform replacement
                    sed -i "s/$old_string/$new_string/g" "$file"
                    # Remove backup if no changes were made
                    if cmp -s "$file" "${file}.bak"; then
                        rm "${file}.bak"
                    fi
                fi
            fi
        fi
    done < <(eval "$find_cmd" -print0)
    
    echo "$total_files:$changed_files"
}

# Function to show summary
show_summary() {
    local total_files="$1"
    local changed_files="$2"
    local backup_dir="$3"
    local dry_run="$4"
    
    echo
    if [ "$dry_run" = "true" ]; then
        print_info "DRY RUN SUMMARY:"
        print_info "Total source files found: $total_files"
        print_info "Files that would be changed: $changed_files"
        print_info "No actual changes were made"
    else
        print_success "REPLACEMENT SUMMARY:"
        print_success "Total source files processed: $total_files"
        print_success "Files changed: $changed_files"
        if [ -n "$backup_dir" ]; then
            print_success "Backup created in: $backup_dir"
        fi
    fi
}

# Main script
main() {
    # Parse command line arguments
    local old_string=""
    local new_string=""
    local target_dir="."
    local recursive=false
    local create_backup_flag=false
    local dry_run=false
    
    # Parse options
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -r|--recursive)
                recursive=true
                shift
                ;;
            -b|--backup)
                create_backup_flag=true
                shift
                ;;
            -n|--dry-run)
                dry_run=true
                create_backup_flag=false
                shift
                ;;
            -*)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
            *)
                if [ -z "$old_string" ]; then
                    old_string="$1"
                elif [ -z "$new_string" ]; then
                    new_string="$1"
                elif [ -z "$target_dir" ]; then
                    target_dir="$1"
                else
                    print_error "Too many arguments"
                    show_usage
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    # Validate arguments
    validate_args "$old_string" "$new_string"
    
    # Check directory
    check_directory "$target_dir"
    
    # Show operation summary
    print_info "Operation Summary:"
    print_info "  Old string: '$old_string'"
    print_info "  New string: '$new_string'"
    print_info "  Target directory: $target_dir"
    print_info "  Recursive: $recursive"
    print_info "  Dry run: $dry_run"
    print_info "  Create backup: $create_backup_flag"
    
    if [ "$dry_run" = "false" ]; then
        echo
        echo "Are you sure you want to proceed? (y/N)"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            print_info "Operation aborted by user"
            exit 0
        fi
    fi
    
    # Create backup if needed
    local backup_dir=""
    if [ "$create_backup_flag" = "true" ] && [ "$dry_run" = "false" ]; then
        backup_dir=$(create_backup "$target_dir")
    fi
    
    # Perform replacement
    local result
    result=$(perform_replacement "$target_dir" "$old_string" "$new_string" "$dry_run" "$recursive")
    local total_files="${result%:*}"
    local changed_files="${result#*:}"
    
    # Show summary
    show_summary "$total_files" "$changed_files" "$backup_dir" "$dry_run"
    
    if [ "$dry_run" = "false" ] && [ "$changed_files" -gt 0 ]; then
        print_success "String replacement completed successfully!"
    fi
}

# Run main function with all arguments
main "$@"
