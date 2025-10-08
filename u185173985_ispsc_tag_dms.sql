-- phpMyAdmin SQL Dump
-- version 5.2.2
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1:3306
-- Generation Time: Oct 08, 2025 at 06:25 AM
-- Server version: 11.8.3-MariaDB-log
-- PHP Version: 7.2.34

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `u185173985_ispsc_tag_dms`
--

-- --------------------------------------------------------

--
-- Table structure for table `action_required`
--

CREATE TABLE `action_required` (
  `action_id` int(11) NOT NULL,
  `action_name` varchar(255) NOT NULL,
  `action_description` text DEFAULT NULL,
  `action_category` enum('decision','communication','document_management','administrative','custom') DEFAULT 'custom',
  `is_active` tinyint(1) DEFAULT 1,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  `created_by` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `action_required`
--

INSERT INTO `action_required` (`action_id`, `action_name`, `action_description`, `action_category`, `is_active`, `created_at`, `updated_at`, `created_by`) VALUES
(1, 'Appropriate Action', 'Take appropriate action based on document content', 'decision', 1, '2024-12-31 16:00:00', '2024-12-31 16:00:00', 1),
(2, 'Study/Information', 'Study the document for information purposes', 'decision', 1, '2024-12-31 16:00:00', '2024-12-31 16:00:00', 1),
(3, 'Signature', 'Document requires signature', 'decision', 1, '2024-12-31 16:00:00', '2024-12-31 16:00:00', 1),
(4, 'Clearance/Approval', 'Document requires clearance or approval', 'decision', 1, '2024-12-31 16:00:00', '2024-12-31 16:00:00', 1),
(5, 'Comments/Feedback', 'Provide comments or feedback on the document', 'communication', 1, '2024-12-31 16:00:00', '2024-12-31 16:00:00', 1),
(6, 'Reply', 'Reply to the document or sender', 'communication', 1, '2024-12-31 16:00:00', '2024-12-31 16:00:00', 1),
(7, 'Take up with me', 'Discuss the document in person', 'communication', 1, '2024-12-31 16:00:00', '2024-12-31 16:00:00', 1),
(8, 'File', 'File the document for record keeping', 'document_management', 1, '2024-12-31 16:00:00', '2024-12-31 16:00:00', 1),
(9, 'Return', 'Return the document to sender', 'document_management', 1, '2024-12-31 16:00:00', '2024-12-31 16:00:00', 1),
(10, 'Submit Report', 'Submit a report based on the document', 'document_management', 1, '2024-12-31 16:00:00', '2024-12-31 16:00:00', 1),
(11, 'Submit CSW', 'Submit CSW (Complete Staff Work)', 'document_management', 1, '2024-12-31 16:00:00', '2024-12-31 16:00:00', 1),
(12, 'Attendance', 'Mark attendance or participation', 'administrative', 1, '2024-12-31 16:00:00', '2024-12-31 16:00:00', 1),
(13, 'Review', 'Review the document thoroughly', 'decision', 1, '2024-12-31 16:00:00', '2024-12-31 16:00:00', 1),
(14, 'Prepare Memo/Special', 'Prepare a memo or special order', 'document_management', 1, '2024-12-31 16:00:00', '2024-12-31 16:00:00', 1),
(15, 'Order/Office Order', 'Issue an order or office order', 'document_management', 1, '2024-12-31 16:00:00', '2024-12-31 16:00:00', 1),
(16, 'Representation', 'Handle representation on behalf of', 'administrative', 1, '2024-12-31 16:00:00', '2024-12-31 16:00:00', 1);

-- --------------------------------------------------------

--
-- Table structure for table `announcements`
--

CREATE TABLE `announcements` (
  `announcement_id` int(11) NOT NULL,
  `title` varchar(255) NOT NULL,
  `body` text NOT NULL,
  `visible_to_all` tinyint(1) DEFAULT 1,
  `status` enum('draft','scheduled','published','archived') DEFAULT 'draft',
  `publish_at` datetime DEFAULT NULL,
  `expire_at` datetime DEFAULT NULL,
  `created_by_name` varchar(150) DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `announcement_departments`
--

CREATE TABLE `announcement_departments` (
  `announcement_id` int(11) NOT NULL,
  `department_id` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `announcement_roles`
--

CREATE TABLE `announcement_roles` (
  `announcement_id` int(11) NOT NULL,
  `role` enum('ADMIN','DEAN','FACULTY') NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `announcement_users`
--

CREATE TABLE `announcement_users` (
  `announcement_id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `departments`
--

CREATE TABLE `departments` (
  `department_id` int(11) NOT NULL,
  `name` varchar(100) NOT NULL,
  `code` varchar(20) DEFAULT NULL,
  `is_active` tinyint(1) DEFAULT 1,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `departments`
--

INSERT INTO `departments` (`department_id`, `name`, `code`, `is_active`, `created_at`, `updated_at`) VALUES
(1, 'College of Arts and Sciences', 'CAS', 1, '2025-09-30 01:29:43', '2025-09-30 01:29:43'),
(2, 'College of Management and Business Economics', 'CMBE', 1, '2025-09-30 01:29:43', '2025-09-30 01:29:43'),
(3, 'College of Teacher Education', 'CTE', 1, '2025-09-30 01:29:43', '2025-09-30 01:29:43'),
(4, 'Laboratory High School', 'LHS', 1, '2025-09-30 01:29:43', '2025-09-30 01:29:43'),
(6, 'Graduate School', 'GRADUATE SCHOOL', 1, '2025-09-30 01:29:43', '2025-09-30 01:29:43'),
(999, 'Unassigned', 'UNASSIGNED', 1, '2025-09-30 02:34:44', '2025-09-30 02:34:44'),
(1000, 'Developer', 'TECH', 1, '2025-10-05 22:54:14', '2025-10-05 22:54:14');

-- --------------------------------------------------------

--
-- Table structure for table `dms_documents`
--

CREATE TABLE `dms_documents` (
  `doc_id` int(11) NOT NULL,
  `doc_type` int(11) NOT NULL,
  `folder_id` int(11) DEFAULT NULL,
  `reference` varchar(255) DEFAULT NULL,
  `title` varchar(255) NOT NULL,
  `revision` varchar(50) DEFAULT NULL,
  `rev_date` date DEFAULT NULL,
  `from_field` varchar(255) DEFAULT NULL,
  `to_field` varchar(255) DEFAULT NULL,
  `date_received` date DEFAULT NULL,
  `google_drive_link` text DEFAULT NULL,
  `description` text DEFAULT NULL,
  `available_copy` enum('hard_copy','soft_copy','both') DEFAULT 'soft_copy',
  `received_by` varchar(255) DEFAULT NULL,
  `received_by_user_id` int(11) DEFAULT NULL,
  `visible_to_all` tinyint(1) DEFAULT 1,
  `allowed_user_ids` text DEFAULT NULL,
  `allowed_roles` text DEFAULT NULL,
  `status` enum('active','inactive','pending','deleted') DEFAULT 'active',
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  `created_by_name` varchar(150) DEFAULT NULL,
  `deleted` tinyint(1) DEFAULT 0,
  `deleted_at` datetime DEFAULT NULL,
  `deleted_by_name` varchar(150) DEFAULT NULL,
  `restored_at` datetime DEFAULT NULL,
  `restored_by_name` varchar(150) DEFAULT NULL,
  `is_reply_to_doc_id` int(11) DEFAULT NULL,
  `reply_type` varchar(50) DEFAULT NULL,
  `created_by_user_id` int(11) DEFAULT NULL,
  `visibility` enum('ALL','DEPARTMENT','SPECIFIC_USERS','SPECIFIC_ROLES','ROLE_DEPARTMENT') DEFAULT 'ALL',
  `target_users` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL CHECK (json_valid(`target_users`)),
  `target_roles` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL CHECK (json_valid(`target_roles`)),
  `target_role_dept` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL CHECK (json_valid(`target_role_dept`))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `dms_documents`
--

INSERT INTO `dms_documents` (`doc_id`, `doc_type`, `folder_id`, `reference`, `title`, `revision`, `rev_date`, `from_field`, `to_field`, `date_received`, `google_drive_link`, `description`, `available_copy`, `received_by`, `received_by_user_id`, `visible_to_all`, `allowed_user_ids`, `allowed_roles`, `status`, `created_at`, `updated_at`, `created_by_name`, `deleted`, `deleted_at`, `deleted_by_name`, `restored_at`, `restored_by_name`, `is_reply_to_doc_id`, `reply_type`, `created_by_user_id`, `visibility`, `target_users`, `target_roles`, `target_role_dept`) VALUES
(300, 5, 4, 'CAS-2024-001', 'CAS Faculty Meeting Minutes', NULL, NULL, 'CAS Dean', 'CAS Faculty', '2024-01-15', 'https://drive.google.com/file/d/sample1/view', 'January meeting', 'soft_copy', NULL, NULL, 1, NULL, NULL, 'active', '2024-01-15 08:00:00', '2024-01-15 08:00:00', 'ADMIN LAN', 0, NULL, NULL, NULL, NULL, NULL, NULL, 1, 'DEPARTMENT', NULL, NULL, NULL),
(301, 5, 5, 'CMBE-2024-001', 'CMBE Budget Proposal', NULL, NULL, 'CMBE Dean', 'Finance Office', '2024-02-10', 'https://drive.google.com/file/d/sample2/view', 'Annual budget', 'both', NULL, NULL, 1, NULL, NULL, 'active', '2024-02-10 09:00:00', '2024-02-10 09:00:00', 'ADMIN LAN', 0, NULL, NULL, NULL, NULL, NULL, NULL, 1, 'DEPARTMENT', NULL, NULL, NULL),
(302, 5, 6, 'CTE-2024-001', 'CTE Curriculum Review', NULL, NULL, 'CTE Dean', 'CTE Faculty', '2024-03-05', 'https://drive.google.com/file/d/sample3/view', 'Curriculum update', 'soft_copy', NULL, NULL, 1, NULL, NULL, 'active', '2024-03-05 10:00:00', '2024-03-05 10:00:00', 'ADMIN LAN', 0, NULL, NULL, NULL, NULL, NULL, NULL, 1, 'DEPARTMENT', NULL, NULL, NULL),
(303, 5, 7, 'RES-2024-001', 'Research Grant Application', NULL, NULL, 'Research Office', 'All Faculty', '2024-04-12', 'https://drive.google.com/file/d/sample4/view', 'Grant opportunity', 'soft_copy', NULL, NULL, 1, NULL, NULL, 'active', '2024-04-12 11:00:00', '2024-04-12 11:00:00', 'ADMIN LAN', 0, NULL, NULL, NULL, NULL, NULL, NULL, 1, 'ALL', NULL, NULL, NULL),
(304, 5, 8, 'ADM-2024-002', 'Administrative Order No. 5', NULL, NULL, 'Admin Office', 'All Staff', '2024-05-20', 'https://drive.google.com/file/d/sample5/view', 'New policy', 'hard_copy', NULL, NULL, 1, NULL, NULL, 'active', '2024-05-20 08:30:00', '2024-05-20 08:30:00', 'ADMIN LAN', 0, NULL, NULL, NULL, NULL, NULL, NULL, 1, 'ALL', NULL, NULL, NULL),
(305, 1000, 10, 'TECH-2024-001', 'System Maintenance Notice', NULL, NULL, 'IT Department', 'All Users', '2024-06-15', 'https://drive.google.com/file/d/sample6/view', 'Scheduled maintenance', 'soft_copy', NULL, NULL, 1, NULL, NULL, 'active', '2024-06-15 14:00:00', '2024-06-15 14:00:00', 'THIS IS ME', 0, NULL, NULL, NULL, NULL, NULL, NULL, 1, 'ALL', NULL, NULL, NULL),
(306, 5, 4, 'HR-2024-010', 'Leave Application Guidelines', NULL, NULL, 'HR Office', 'All Employees', '2024-07-08', 'https://drive.google.com/file/d/sample7/view', 'Updated guidelines', 'both', NULL, NULL, 1, NULL, NULL, 'active', '2024-07-08 09:15:00', '2024-07-08 09:15:00', 'ADMIN LAN', 0, NULL, NULL, NULL, NULL, NULL, NULL, 1, 'ALL', NULL, NULL, NULL),
(307, 5, 5, 'SA-2024-005', 'Student Scholarship Program', NULL, NULL, 'Student Affairs', 'Students', '2024-08-22', 'https://drive.google.com/file/d/sample8/view', 'Scholarship info', 'soft_copy', NULL, NULL, 1, NULL, NULL, 'active', '2024-08-22 10:30:00', '2024-08-22 10:30:00', 'ADMIN LAN', 0, NULL, NULL, NULL, NULL, NULL, NULL, 1, 'ALL', NULL, NULL, NULL),
(308, 5, 11, 'LEG-2024-003', 'Legal Compliance Memo', NULL, NULL, 'Legal Office', 'Admin', '2024-09-10', 'https://drive.google.com/file/d/sample9/view', 'Compliance requirements', 'hard_copy', NULL, NULL, 1, NULL, NULL, 'active', '2024-09-10 11:45:00', '2024-09-10 11:45:00', 'ADMIN LAN', 0, NULL, NULL, NULL, NULL, NULL, NULL, 1, 'SPECIFIC_USERS', NULL, NULL, NULL),
(309, 5, 12, 'ARC-2024-001', 'Archive Policy Update', NULL, NULL, 'Archive Office', 'All Depts', '2024-10-05', 'https://drive.google.com/file/d/sample10/view', 'New archiving rules', 'soft_copy', NULL, NULL, 1, NULL, NULL, 'inactive', '2024-10-05 13:00:00', '2024-10-05 13:00:00', 'ADMIN LAN', 0, NULL, NULL, NULL, NULL, NULL, NULL, 1, 'ALL', NULL, NULL, NULL),
(310, 5, 4, 'CAS-2025-002', 'CAS Seminar Invitation', NULL, NULL, 'CAS Dean', 'CAS Faculty', '2025-01-20', 'https://drive.google.com/file/d/sample11/view', 'Seminar details', 'soft_copy', NULL, NULL, 1, NULL, NULL, 'active', '2025-01-20 08:00:00', '2025-01-20 08:00:00', 'ADMIN LAN', 0, NULL, NULL, NULL, NULL, NULL, NULL, 1, 'DEPARTMENT', NULL, NULL, NULL),
(311, 5, 5, 'CMBE-2025-002', 'CMBE Industry Partnership', NULL, NULL, 'CMBE Dean', 'CMBE Faculty', '2025-02-14', 'https://drive.google.com/file/d/sample12/view', 'Partnership program', 'both', NULL, NULL, 1, NULL, NULL, 'active', '2025-02-14 09:30:00', '2025-02-14 09:30:00', 'ADMIN LAN', 0, NULL, NULL, NULL, NULL, NULL, NULL, 1, 'DEPARTMENT', NULL, NULL, NULL),
(312, 5, 6, 'CTE-2025-002', 'CTE Teaching Workshop', NULL, NULL, 'CTE Dean', 'CTE Faculty', '2025-03-18', 'https://drive.google.com/file/d/sample13/view', 'Workshop schedule', 'soft_copy', NULL, NULL, 1, NULL, NULL, 'active', '2025-03-18 10:00:00', '2025-03-18 10:00:00', 'ADMIN LAN', 0, NULL, NULL, NULL, NULL, NULL, NULL, 1, 'DEPARTMENT', NULL, NULL, NULL),
(313, 5, 7, 'RES-2025-002', 'Research Publication Guidelines', NULL, NULL, 'Research Office', 'All Faculty', '2025-04-25', 'https://drive.google.com/file/d/sample14/view', 'Publication guide', 'soft_copy', NULL, NULL, 1, NULL, NULL, 'active', '2025-04-25 11:15:00', '2025-04-25 11:15:00', 'ADMIN LAN', 0, NULL, NULL, NULL, NULL, NULL, NULL, 1, 'ALL', NULL, NULL, NULL),
(314, 5, 8, 'ADM-2025-003', 'Administrative Circular No. 8', NULL, NULL, 'Admin Office', 'All Staff', '2025-05-30', 'https://drive.google.com/file/d/sample15/view', 'Important notice', 'hard_copy', NULL, NULL, 1, NULL, NULL, 'active', '2025-05-30 08:45:00', '2025-05-30 08:45:00', 'ADMIN LAN', 0, NULL, NULL, NULL, NULL, NULL, NULL, 1, 'ALL', NULL, NULL, NULL),
(315, 1000, 10, 'TECH-2025-002', 'Software Update Notification', NULL, NULL, 'IT Department', 'All Users', '2025-06-12', 'https://drive.google.com/file/d/sample16/view', 'System upgrade', 'soft_copy', NULL, NULL, 1, NULL, NULL, 'active', '2025-06-12 14:30:00', '2025-06-12 14:30:00', 'THIS IS ME', 0, NULL, NULL, NULL, NULL, NULL, NULL, 1, 'ALL', NULL, NULL, NULL),
(316, 5, 4, 'HR-2025-011', 'Performance Evaluation Form', NULL, NULL, 'HR Office', 'All Employees', '2025-07-16', 'https://drive.google.com/file/d/sample17/view', 'Annual evaluation', 'both', NULL, NULL, 1, NULL, NULL, 'active', '2025-07-16 09:00:00', '2025-07-16 09:00:00', 'ADMIN LAN', 0, NULL, NULL, NULL, NULL, NULL, NULL, 1, 'ALL', NULL, NULL, NULL),
(317, 5, 5, 'SA-2025-006', 'Student Council Election', NULL, NULL, 'Student Affairs', 'Students', '2025-08-20', 'https://drive.google.com/file/d/sample18/view', 'Election guidelines', 'soft_copy', NULL, NULL, 1, NULL, NULL, 'active', '2025-08-20 10:45:00', '2025-08-20 10:45:00', 'ADMIN LAN', 0, NULL, NULL, NULL, NULL, NULL, NULL, 1, 'ALL', NULL, NULL, NULL),
(318, 5, 11, 'LEG-2025-004', 'Data Privacy Compliance', NULL, NULL, 'Legal Office', 'All Depts', '2025-09-15', 'https://drive.google.com/file/d/sample19/view', 'Privacy policy', 'hard_copy', NULL, NULL, 1, NULL, NULL, 'active', '2025-09-15 11:30:00', '2025-09-15 11:30:00', 'ADMIN LAN', 0, NULL, NULL, NULL, NULL, NULL, NULL, 1, 'ROLE_DEPARTMENT', NULL, NULL, NULL),
(319, 5, 12, 'ARC-2025-002', 'Document Retention Schedule', NULL, NULL, 'Archive Office', 'All Depts', '2025-10-01', 'https://drive.google.com/file/d/sample20/view', 'Retention policy', 'soft_copy', NULL, NULL, 1, NULL, NULL, 'active', '2025-10-01 13:15:00', '2025-10-01 13:15:00', 'ADMIN LAN', 0, NULL, NULL, NULL, NULL, NULL, NULL, 1, 'ALL', NULL, NULL, NULL);

-- --------------------------------------------------------

--
-- Table structure for table `dms_user`
--

CREATE TABLE `dms_user` (
  `user_id` int(11) NOT NULL,
  `profile_pic` longtext DEFAULT NULL,
  `user_email` varchar(255) NOT NULL,
  `password` varchar(255) NOT NULL,
  `Username` varchar(100) NOT NULL,
  `firstname` varchar(100) NOT NULL,
  `lastname` varchar(100) NOT NULL,
  `Contact_number` varchar(20) DEFAULT NULL,
  `department_id` int(11) DEFAULT NULL,
  `role` enum('ADMIN','DEAN','FACULTY') DEFAULT 'FACULTY',
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  `status` enum('active','inactive','banned','pending','deleted') DEFAULT 'pending',
  `is_verified` enum('yes','no') DEFAULT 'no',
  `verification_token` varchar(255) DEFAULT NULL,
  `verification_code` varchar(10) DEFAULT NULL,
  `password_reset_code` varchar(6) DEFAULT NULL,
  `password_reset_expires` datetime DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `dms_user`
--

INSERT INTO `dms_user` (`user_id`, `profile_pic`, `user_email`, `password`, `Username`, `firstname`, `lastname`, `Contact_number`, `department_id`, `role`, `created_at`, `updated_at`, `status`, `is_verified`, `verification_token`, `verification_code`, `password_reset_code`, `password_reset_expires`) VALUES
(1, 'https://lh3.googleusercontent.com/a/ACg8ocIhpD_184y_Edc-9RJod2dRMKXfsKksA7Z1AKaWM-5lc7srqUFH=s96-c', 'polandreiladera03@gmail.com', '$2a$12$WbFadt88HuRbtzSOYx5giu7t6roL6dE68qFCccGUHeHuJT5kaGq06', 'THIS IS ME', 'ADMIN LAN', 'LAN', '09184226085', 1, 'ADMIN', '2025-09-07 03:34:43', '2025-10-06 01:18:10', 'active', 'yes', NULL, NULL, NULL, NULL),
(132, NULL, 'ispscladera@gmail.com', '$2b$12$5J2bEPGSPcIjhTNWLzE0zOyg7NSOrvUaGVrAhsCcSiIhyki584xNm', 'ab111', 'TEST', '1', '09184226085', 1000, 'ADMIN', '2025-10-07 23:04:22', '2025-10-08 00:12:50', 'active', 'yes', NULL, NULL, NULL, NULL),
(133, 'https://lh3.googleusercontent.com/a/ACg8ocInYH-jR1J2gXLgGoO3kFnt0kUsWjBRw-4Qr_CNfj8KklKV5GM=s96-c', 'kaelgiongan@gmail.com', '', 'Test account', 'John', 'Michael', '09177428469', 1, 'DEAN', '2025-10-08 00:09:03', '2025-10-08 00:29:10', 'active', 'yes', NULL, NULL, NULL, NULL),
(134, 'https://lh3.googleusercontent.com/a/ACg8ocKY7FHsiOFJd8h-754VQgb1muTsSXKonCDewIZtAZ5GR1itVQ=s96-c', 'capstoneuser101@gmail.com', '', 'capstone 101', '', '', NULL, NULL, 'FACULTY', '2025-10-08 05:22:45', '2025-10-08 05:22:45', 'active', 'yes', NULL, NULL, NULL, NULL),
(135, 'https://lh3.googleusercontent.com/a/ACg8ocKf72bFG-I8hFWQJT_IldNd88VfUcOvU6W8dLHEW809kL8rhtfJ=s96-c', 'merinpiolo9@gmail.com', '', 'Piolo Merin', '', '', NULL, NULL, 'FACULTY', '2025-10-08 05:23:22', '2025-10-08 05:23:22', 'active', 'yes', NULL, NULL, NULL, NULL),
(136, 'https://lh3.googleusercontent.com/a/ACg8ocLV-XDgqUWjRyS9kVC_pJNkjOG7n6LJVcVeD7mWS9FDJ30CerUs=s96-c', 'mainlan03@gmail.com', '', 'Admin Lan', '', '', NULL, NULL, 'FACULTY', '2025-10-08 05:24:38', '2025-10-08 05:24:38', 'active', 'yes', NULL, NULL, NULL, NULL);

--
-- Triggers `dms_user`
--
DELIMITER $$
CREATE TRIGGER `set_first_user_as_admin` BEFORE INSERT ON `dms_user` FOR EACH ROW BEGIN
    DECLARE user_count INT DEFAULT 0;
    
    -- Check if this is the first user being created
    SELECT COUNT(*) INTO user_count FROM dms_user;
    
    -- If this is the first user (user_count = 0), make them ADMIN
    IF user_count = 0 THEN
        SET NEW.role = 'ADMIN';
        SET NEW.is_verified = 'yes';
        SET NEW.status = 'active';
    END IF;
END
$$
DELIMITER ;
DELIMITER $$
CREATE TRIGGER `set_initial_user_status` BEFORE INSERT ON `dms_user` FOR EACH ROW BEGIN
    IF NEW.is_verified = 'yes' THEN
        SET NEW.status = 'active';
    ELSE
        SET NEW.status = 'pending';
    END IF;
END
$$
DELIMITER ;
DELIMITER $$
CREATE TRIGGER `update_user_status_on_verification` BEFORE UPDATE ON `dms_user` FOR EACH ROW BEGIN
    -- Only update status if is_verified is being changed and status is not being explicitly set
    IF NEW.is_verified != OLD.is_verified THEN
        IF NEW.is_verified = 'yes' AND OLD.is_verified = 'no' THEN
            SET NEW.status = 'active';
        ELSEIF NEW.is_verified = 'no' AND OLD.is_verified = 'yes' THEN
            SET NEW.status = 'pending';
        END IF;
    END IF;
END
$$
DELIMITER ;

-- --------------------------------------------------------

--
-- Table structure for table `document_actions`
--

CREATE TABLE `document_actions` (
  `document_action_id` int(11) NOT NULL,
  `doc_id` int(11) NOT NULL,
  `action_id` int(11) NOT NULL,
  `assigned_to_user_id` int(11) DEFAULT NULL,
  `assigned_to_role` enum('ADMIN','DEAN','FACULTY') DEFAULT NULL,
  `assigned_to_department_id` int(11) DEFAULT NULL,
  `status` enum('pending','in_progress','completed','cancelled') DEFAULT 'pending',
  `priority` enum('low','medium','high','urgent') DEFAULT 'medium',
  `due_date` date DEFAULT NULL,
  `completed_at` datetime DEFAULT NULL,
  `completed_by_user_id` int(11) DEFAULT NULL,
  `notes` text DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  `created_by_user_id` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `document_actions`
--

INSERT INTO `document_actions` (`document_action_id`, `doc_id`, `action_id`, `assigned_to_user_id`, `assigned_to_role`, `assigned_to_department_id`, `status`, `priority`, `due_date`, `completed_at`, `completed_by_user_id`, `notes`, `created_at`, `updated_at`, `created_by_user_id`) VALUES
(33, 213, 3, 133, NULL, NULL, 'pending', 'medium', NULL, NULL, NULL, NULL, '2025-10-08 00:46:13', '2025-10-08 00:46:13', 1);

-- --------------------------------------------------------

--
-- Table structure for table `document_departments`
--

CREATE TABLE `document_departments` (
  `doc_id` int(11) NOT NULL,
  `department_id` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `document_departments`
--

INSERT INTO `document_departments` (`doc_id`, `department_id`) VALUES
(131, 999),
(211, 999),
(212, 1);

-- --------------------------------------------------------

--
-- Table structure for table `document_folders`
--

CREATE TABLE `document_folders` (
  `id` int(11) NOT NULL,
  `doc_id` int(11) NOT NULL,
  `folder_id` int(11) NOT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `document_folders`
--

INSERT INTO `document_folders` (`id`, `doc_id`, `folder_id`, `created_at`) VALUES
(57, 123, 12, '2025-10-07 01:13:34'),
(58, 115, 8, '2025-10-07 02:45:12'),
(59, 125, 999, '2025-10-07 02:47:11'),
(60, 126, 999, '2025-10-07 02:48:59'),
(61, 128, 7, '2025-10-07 07:50:20'),
(62, 131, 999, '2025-10-08 00:21:35'),
(63, 211, 999, '2025-10-08 00:29:30'),
(64, 212, 10, '2025-10-08 00:31:58');

-- --------------------------------------------------------

--
-- Table structure for table `document_roles`
--

CREATE TABLE `document_roles` (
  `doc_id` int(11) NOT NULL,
  `role` enum('ADMIN','DEAN','FACULTY') NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `document_types`
--

CREATE TABLE `document_types` (
  `type_id` int(11) NOT NULL,
  `name` varchar(100) NOT NULL,
  `description` text DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `document_types`
--

INSERT INTO `document_types` (`type_id`, `name`, `description`, `created_at`, `updated_at`) VALUES
(5, 'Letter', NULL, '2025-09-29 00:02:46', '2025-09-29 00:02:46'),
(999, 'Unassigned', 'Default type for documents with deleted document types', '2025-09-30 02:34:44', '2025-09-30 02:34:44'),
(1000, 'Trial', '', '2025-10-05 10:56:42', '2025-10-05 10:56:42');

-- --------------------------------------------------------

--
-- Table structure for table `document_users`
--

CREATE TABLE `document_users` (
  `doc_id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `folders`
--

CREATE TABLE `folders` (
  `folder_id` int(11) NOT NULL,
  `name` varchar(100) NOT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `folders`
--

INSERT INTO `folders` (`folder_id`, `name`, `created_at`, `updated_at`) VALUES
(4, 'Human Resources', '2025-09-23 01:48:31', '2025-09-23 01:48:31'),
(5, 'Student Affairs', '2025-09-23 01:48:31', '2025-09-23 01:48:31'),
(6, 'Faculty Development', '2025-09-23 01:48:31', '2025-09-23 01:48:31'),
(7, 'Research and Extension', '2025-09-23 01:48:31', '2025-09-23 01:48:31'),
(8, 'Administrative Orders', '2025-09-23 01:48:31', '2025-09-23 01:48:31'),
(10, 'Procurement', '2025-09-23 01:48:31', '2025-09-23 01:48:31'),
(11, 'Legal Documents', '2025-09-23 01:48:31', '2025-09-23 01:48:31'),
(12, 'Archived Records', '2025-09-23 01:48:31', '2025-09-23 01:48:31'),
(999, 'Unassigned', '2025-09-30 02:34:44', '2025-09-30 02:34:44');

-- --------------------------------------------------------

--
-- Table structure for table `notifications`
--

CREATE TABLE `notifications` (
  `notification_id` int(11) NOT NULL,
  `title` varchar(255) NOT NULL,
  `message` text NOT NULL,
  `type` enum('added','updated','deleted','requested','replied') NOT NULL,
  `visible_to_all` tinyint(1) DEFAULT 1,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `related_doc_id` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `notifications`
--

INSERT INTO `notifications` (`notification_id`, `title`, `message`, `type`, `visible_to_all`, `created_at`, `related_doc_id`) VALUES
(445, 'New User Registration: John Michael Giong-an', 'A new user \"John Michael Giong-an\" (kaelgiongan@gmail.com) has registered via Google OAuth.', '', 0, '2025-10-08 00:09:03', NULL),
(446, 'New Document Added: Pic', 'A new document \"Pic\" has been uploaded by Test account', 'added', 0, '2025-10-08 00:21:35', 131),
(447, 'New Document Added: signature', 'A new document \"signature\" has been uploaded by Test account', 'added', 0, '2025-10-08 00:29:30', 211),
(448, 'New Document Added: 1111', 'A new document \"1111\" has been uploaded by THIS IS ME', 'added', 0, '2025-10-08 00:31:58', 212),
(449, 'Request Added: 13235', 'You have a new request \"13235\" from THIS IS ME', 'requested', 0, '2025-10-08 00:46:13', 213),
(450, 'New User Registration: capstone 101', 'A new user \"capstone 101\" (capstoneuser101@gmail.com) has registered via Google OAuth.', '', 0, '2025-10-08 05:22:45', NULL),
(451, 'New User Registration: Piolo Merin', 'A new user \"Piolo Merin\" (merinpiolo9@gmail.com) has registered via Google OAuth.', '', 0, '2025-10-08 05:23:22', NULL),
(452, 'New User Registration: Admin Lan', 'A new user \"Admin Lan\" (mainlan03@gmail.com) has registered via Google OAuth.', '', 0, '2025-10-08 05:24:38', NULL);

-- --------------------------------------------------------

--
-- Table structure for table `notification_departments`
--

CREATE TABLE `notification_departments` (
  `notification_id` int(11) NOT NULL,
  `department_id` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `notification_departments`
--

INSERT INTO `notification_departments` (`notification_id`, `department_id`) VALUES
(448, 1),
(446, 999),
(447, 999);

-- --------------------------------------------------------

--
-- Table structure for table `notification_reads`
--

CREATE TABLE `notification_reads` (
  `notification_id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL,
  `read_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `notification_reads`
--

INSERT INTO `notification_reads` (`notification_id`, `user_id`, `read_at`) VALUES
(449, 133, '2025-10-08 05:30:10');

-- --------------------------------------------------------

--
-- Table structure for table `notification_roles`
--

CREATE TABLE `notification_roles` (
  `notification_id` int(11) NOT NULL,
  `role` enum('ADMIN','DEAN','FACULTY') NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `notification_roles`
--

INSERT INTO `notification_roles` (`notification_id`, `role`) VALUES
(446, 'ADMIN'),
(447, 'ADMIN');

-- --------------------------------------------------------

--
-- Table structure for table `notification_users`
--

CREATE TABLE `notification_users` (
  `notification_id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `notification_users`
--

INSERT INTO `notification_users` (`notification_id`, `user_id`) VALUES
(445, 1),
(446, 1),
(450, 1),
(451, 1),
(452, 1),
(450, 132),
(451, 132),
(452, 132),
(449, 133),
(450, 133),
(451, 133),
(452, 133);

-- --------------------------------------------------------

--
-- Table structure for table `others`
--

CREATE TABLE `others` (
  `other_id` int(11) NOT NULL,
  `other_name` varchar(255) NOT NULL,
  `category` enum('ICON','MANUAL','POLICY','TERMS','INFO') NOT NULL,
  `link` varchar(500) DEFAULT NULL,
  `created_by_user_id` int(11) DEFAULT NULL,
  `updated_by_user_id` int(11) DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `others`
--

INSERT INTO `others` (`other_id`, `other_name`, `category`, `link`, `created_by_user_id`, `updated_by_user_id`, `created_at`, `updated_at`) VALUES
(1, 'ICON_1', 'ICON', 'https://i.imgur.com/YeJZBRK.png', NULL, NULL, '2025-10-05 23:57:16', '2025-10-06 00:57:11'),
(2, 'ICON_2', 'ICON', 'https://i.imgur.com/rthDwK6.png', NULL, NULL, '2025-10-05 23:57:16', '2025-10-06 00:56:12'),
(3, 'ICON_3', 'ICON', 'https://i.imgur.com/IJ0XY4O.png', NULL, NULL, '2025-10-05 23:57:16', '2025-10-06 00:56:12'),
(4, 'ICON_4', 'ICON', 'https://i.imgur.com/9dAHRk6.png', NULL, NULL, '2025-10-05 23:57:16', '2025-10-06 00:56:12'),
(5, 'ICON_5', 'ICON', 'https://i.imgur.com/qVlV7PQ.png', NULL, NULL, '2025-10-05 23:57:16', '2025-10-06 00:56:12'),
(6, 'ICON_6', 'ICON', 'https://i.imgur.com/esoqCrY.png', NULL, NULL, '2025-10-05 23:57:16', '2025-10-06 00:56:12'),
(7, 'ICON_7', 'ICON', 'https://i.imgur.com/r5F2yu6.png', NULL, NULL, '2025-10-05 23:57:16', '2025-10-06 00:56:12'),
(8, 'ICON_8', 'ICON', 'https://i.imgur.com/CjyQ5O0.png', NULL, NULL, '2025-10-05 23:57:16', '2025-10-06 00:56:12'),
(9, 'ICON_9', 'ICON', 'https://i.imgur.com/hqw5Nst.png', NULL, NULL, '2025-10-05 23:57:16', '2025-10-06 00:56:12'),
(10, 'ICON_10', 'ICON', 'https://i.imgur.com/zegmH78.png', NULL, NULL, '2025-10-05 23:57:16', '2025-10-06 00:56:12'),
(11, 'ICON_11', 'ICON', 'https://i.imgur.com/OdNn6V3.png', NULL, NULL, '2025-10-05 23:57:16', '2025-10-06 00:56:12'),
(12, 'ICON_12', 'ICON', 'https://i.imgur.com/ZMhF0SD.png', NULL, NULL, '2025-10-05 23:57:16', '2025-10-06 00:56:12'),
(13, 'ICON_13', 'ICON', 'https://i.imgur.com/MnE7tkW.png', NULL, NULL, '2025-10-05 23:57:16', '2025-10-06 00:56:12'),
(14, 'ICON_14', 'ICON', 'https://i.imgur.com/GdufXPv.png', NULL, NULL, '2025-10-05 23:57:16', '2025-10-06 00:56:12'),
(15, 'USER & MAINTENANCE MANUAL', 'MANUAL', 'https://docs.google.com/document/d/e/2PACX-1vQnSUQBOQjYw_J0Jx1QLVTGH0KKxWOn4_XXvFkYjQhJ5EY4uUsRtVeXw1OlE7xvFw/pub', NULL, NULL, '2025-10-05 23:57:16', '2025-10-06 01:19:36'),
(16, 'PRIVACY POLICY', 'POLICY', 'https://docs.google.com/document/d/176P3Zt5Oa57zzcK_L41Yf7oL0GFLGYWaGaFGJg3l56o/edit?usp=sharing', NULL, NULL, '2025-10-05 23:57:16', '2025-10-05 23:57:16'),
(17, 'TERMS & CONDITIONS', 'TERMS', 'https://docs.google.com/document/d/16NjJI5X69m0wcCv33NK5VP2oEm3rnqVwDoQqfx_ZZn0/edit?usp=drive_link', NULL, NULL, '2025-10-05 23:57:16', '2025-10-05 23:57:16'),
(18, 'CONTACT', 'INFO', 'polandreiladera03@gmail.com', NULL, NULL, '2025-10-05 23:57:16', '2025-10-05 23:57:16'),
(19, 'FACEBOOK', 'INFO', 'https://facebook.com/ispsc', NULL, NULL, '2025-10-06 02:37:39', '2025-10-06 02:37:39'),
(20, 'TWITTER', 'INFO', 'https://twitter.com/ispsc', NULL, NULL, '2025-10-06 02:37:39', '2025-10-06 02:37:39'),
(21, 'INSTAGRAM', 'INFO', 'https://instagram.com/ispsc', NULL, NULL, '2025-10-06 02:37:39', '2025-10-06 02:37:39');

-- --------------------------------------------------------

--
-- Table structure for table `system_settings`
--

CREATE TABLE `system_settings` (
  `id` int(11) NOT NULL,
  `maintenance_mode` tinyint(1) NOT NULL DEFAULT 0,
  `maintenance_message` text DEFAULT NULL,
  `maintenance_start_time` datetime DEFAULT NULL,
  `maintenance_end_time` datetime DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `system_settings`
--

INSERT INTO `system_settings` (`id`, `maintenance_mode`, `maintenance_message`, `maintenance_start_time`, `maintenance_end_time`, `created_at`, `updated_at`) VALUES
(1, 0, NULL, NULL, NULL, '2025-10-06 09:28:56', '2025-10-07 02:52:56');

-- --------------------------------------------------------

--
-- Table structure for table `user_document_preferences`
--

CREATE TABLE `user_document_preferences` (
  `preference_id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL,
  `doc_id` int(11) NOT NULL,
  `is_favorite` tinyint(1) DEFAULT 0,
  `is_pinned` tinyint(1) DEFAULT 0,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Indexes for dumped tables
--

--
-- Indexes for table `action_required`
--
ALTER TABLE `action_required`
  ADD PRIMARY KEY (`action_id`),
  ADD KEY `idx_action_category` (`action_category`),
  ADD KEY `idx_is_active` (`is_active`),
  ADD KEY `idx_created_by` (`created_by`);

--
-- Indexes for table `announcements`
--
ALTER TABLE `announcements`
  ADD PRIMARY KEY (`announcement_id`),
  ADD KEY `idx_status` (`status`),
  ADD KEY `idx_publish_at` (`publish_at`),
  ADD KEY `idx_expire_at` (`expire_at`);

--
-- Indexes for table `announcement_departments`
--
ALTER TABLE `announcement_departments`
  ADD PRIMARY KEY (`announcement_id`,`department_id`),
  ADD KEY `department_id` (`department_id`);

--
-- Indexes for table `announcement_roles`
--
ALTER TABLE `announcement_roles`
  ADD PRIMARY KEY (`announcement_id`,`role`),
  ADD KEY `role` (`role`);

--
-- Indexes for table `announcement_users`
--
ALTER TABLE `announcement_users`
  ADD PRIMARY KEY (`announcement_id`,`user_id`),
  ADD KEY `user_id` (`user_id`);

--
-- Indexes for table `departments`
--
ALTER TABLE `departments`
  ADD PRIMARY KEY (`department_id`),
  ADD UNIQUE KEY `name` (`name`),
  ADD UNIQUE KEY `code` (`code`);

--
-- Indexes for table `dms_documents`
--
ALTER TABLE `dms_documents`
  ADD PRIMARY KEY (`doc_id`),
  ADD KEY `idx_doc_type` (`doc_type`),
  ADD KEY `idx_title` (`title`),
  ADD KEY `idx_available_copy` (`available_copy`),
  ADD KEY `idx_received_by_user_id` (`received_by_user_id`),
  ADD KEY `idx_visible_to_all` (`visible_to_all`),
  ADD KEY `idx_status` (`status`),
  ADD KEY `idx_deleted` (`deleted`),
  ADD KEY `idx_deleted_at` (`deleted_at`),
  ADD KEY `idx_deleted_by_name` (`deleted_by_name`),
  ADD KEY `dms_documents_ibfk_reply` (`is_reply_to_doc_id`),
  ADD KEY `idx_folder_id` (`folder_id`),
  ADD KEY `dms_documents_ibfk_created_by` (`created_by_user_id`);

--
-- Indexes for table `dms_user`
--
ALTER TABLE `dms_user`
  ADD PRIMARY KEY (`user_id`),
  ADD KEY `dms_user_ibfk_department` (`department_id`),
  ADD KEY `idx_password_reset` (`user_email`,`password_reset_code`,`password_reset_expires`);

--
-- Indexes for table `document_actions`
--
ALTER TABLE `document_actions`
  ADD PRIMARY KEY (`document_action_id`),
  ADD KEY `idx_doc_id` (`doc_id`),
  ADD KEY `idx_action_id` (`action_id`),
  ADD KEY `idx_assigned_to_user_id` (`assigned_to_user_id`),
  ADD KEY `idx_assigned_to_role` (`assigned_to_role`),
  ADD KEY `idx_assigned_to_department_id` (`assigned_to_department_id`),
  ADD KEY `idx_status` (`status`),
  ADD KEY `idx_priority` (`priority`),
  ADD KEY `idx_due_date` (`due_date`),
  ADD KEY `idx_completed_by_user_id` (`completed_by_user_id`),
  ADD KEY `idx_created_by_user_id` (`created_by_user_id`);

--
-- Indexes for table `document_departments`
--
ALTER TABLE `document_departments`
  ADD KEY `document_departments_ibfk_doc` (`doc_id`),
  ADD KEY `document_departments_ibfk_department` (`department_id`);

--
-- Indexes for table `document_folders`
--
ALTER TABLE `document_folders`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `uniq_doc_folder_pair` (`doc_id`,`folder_id`),
  ADD KEY `idx_doc` (`doc_id`),
  ADD KEY `idx_folder` (`folder_id`);

--
-- Indexes for table `document_roles`
--
ALTER TABLE `document_roles`
  ADD KEY `document_roles_ibfk_doc` (`doc_id`);

--
-- Indexes for table `document_types`
--
ALTER TABLE `document_types`
  ADD PRIMARY KEY (`type_id`),
  ADD UNIQUE KEY `uniq_document_types_name` (`name`);

--
-- Indexes for table `document_users`
--
ALTER TABLE `document_users`
  ADD KEY `document_users_ibfk_doc` (`doc_id`),
  ADD KEY `document_users_ibfk_user` (`user_id`);

--
-- Indexes for table `folders`
--
ALTER TABLE `folders`
  ADD PRIMARY KEY (`folder_id`),
  ADD UNIQUE KEY `name` (`name`),
  ADD UNIQUE KEY `uniq_folders_name` (`name`);

--
-- Indexes for table `notifications`
--
ALTER TABLE `notifications`
  ADD PRIMARY KEY (`notification_id`),
  ADD KEY `idx_type` (`type`),
  ADD KEY `idx_visible_to_all` (`visible_to_all`),
  ADD KEY `idx_created_at` (`created_at`),
  ADD KEY `idx_related_doc_id` (`related_doc_id`);

--
-- Indexes for table `notification_departments`
--
ALTER TABLE `notification_departments`
  ADD PRIMARY KEY (`notification_id`,`department_id`),
  ADD KEY `department_id` (`department_id`),
  ADD KEY `idx_nd_notif` (`notification_id`),
  ADD KEY `idx_nd_dept` (`department_id`);

--
-- Indexes for table `notification_reads`
--
ALTER TABLE `notification_reads`
  ADD PRIMARY KEY (`notification_id`,`user_id`),
  ADD UNIQUE KEY `uq_notification_reads` (`notification_id`,`user_id`),
  ADD KEY `idx_user_id` (`user_id`);

--
-- Indexes for table `notification_roles`
--
ALTER TABLE `notification_roles`
  ADD PRIMARY KEY (`notification_id`,`role`),
  ADD KEY `role` (`role`),
  ADD KEY `idx_nr_notif` (`notification_id`),
  ADD KEY `idx_nr_role` (`role`);

--
-- Indexes for table `notification_users`
--
ALTER TABLE `notification_users`
  ADD PRIMARY KEY (`notification_id`,`user_id`),
  ADD KEY `user_id` (`user_id`),
  ADD KEY `idx_nu_notif` (`notification_id`),
  ADD KEY `idx_nu_user` (`user_id`);

--
-- Indexes for table `others`
--
ALTER TABLE `others`
  ADD PRIMARY KEY (`other_id`),
  ADD UNIQUE KEY `uq_others_category_name` (`category`,`other_name`),
  ADD KEY `idx_category` (`category`),
  ADD KEY `fk_others_created_by` (`created_by_user_id`),
  ADD KEY `fk_others_updated_by` (`updated_by_user_id`);

--
-- Indexes for table `system_settings`
--
ALTER TABLE `system_settings`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `user_document_preferences`
--
ALTER TABLE `user_document_preferences`
  ADD PRIMARY KEY (`preference_id`),
  ADD UNIQUE KEY `uniq_user_doc_pref` (`user_id`,`doc_id`),
  ADD KEY `user_id` (`user_id`),
  ADD KEY `doc_id` (`doc_id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `action_required`
--
ALTER TABLE `action_required`
  MODIFY `action_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=18;

--
-- AUTO_INCREMENT for table `announcements`
--
ALTER TABLE `announcements`
  MODIFY `announcement_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=48;

--
-- AUTO_INCREMENT for table `departments`
--
ALTER TABLE `departments`
  MODIFY `department_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=1001;

--
-- AUTO_INCREMENT for table `dms_documents`
--
ALTER TABLE `dms_documents`
  MODIFY `doc_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=214;

--
-- AUTO_INCREMENT for table `dms_user`
--
ALTER TABLE `dms_user`
  MODIFY `user_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=137;

--
-- AUTO_INCREMENT for table `document_actions`
--
ALTER TABLE `document_actions`
  MODIFY `document_action_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=34;

--
-- AUTO_INCREMENT for table `document_folders`
--
ALTER TABLE `document_folders`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=65;

--
-- AUTO_INCREMENT for table `document_types`
--
ALTER TABLE `document_types`
  MODIFY `type_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=1001;

--
-- AUTO_INCREMENT for table `folders`
--
ALTER TABLE `folders`
  MODIFY `folder_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=1001;

--
-- AUTO_INCREMENT for table `notifications`
--
ALTER TABLE `notifications`
  MODIFY `notification_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=453;

--
-- AUTO_INCREMENT for table `others`
--
ALTER TABLE `others`
  MODIFY `other_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=22;

--
-- AUTO_INCREMENT for table `user_document_preferences`
--
ALTER TABLE `user_document_preferences`
  MODIFY `preference_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=27;

--
-- Constraints for dumped tables
--

--
-- Constraints for table `announcement_departments`
--
ALTER TABLE `announcement_departments`
  ADD CONSTRAINT `announcement_departments_ibfk_announcement` FOREIGN KEY (`announcement_id`) REFERENCES `announcements` (`announcement_id`) ON DELETE CASCADE,
  ADD CONSTRAINT `announcement_departments_ibfk_department` FOREIGN KEY (`department_id`) REFERENCES `departments` (`department_id`) ON DELETE CASCADE;

--
-- Constraints for table `announcement_roles`
--
ALTER TABLE `announcement_roles`
  ADD CONSTRAINT `announcement_roles_ibfk_announcement` FOREIGN KEY (`announcement_id`) REFERENCES `announcements` (`announcement_id`) ON DELETE CASCADE;

--
-- Constraints for table `announcement_users`
--
ALTER TABLE `announcement_users`
  ADD CONSTRAINT `announcement_users_ibfk_announcement` FOREIGN KEY (`announcement_id`) REFERENCES `announcements` (`announcement_id`) ON DELETE CASCADE,
  ADD CONSTRAINT `announcement_users_ibfk_user` FOREIGN KEY (`user_id`) REFERENCES `dms_user` (`user_id`) ON DELETE CASCADE;

--
-- Constraints for table `dms_documents`
--
ALTER TABLE `dms_documents`
  ADD CONSTRAINT `dms_documents_ibfk_created_by` FOREIGN KEY (`created_by_user_id`) REFERENCES `dms_user` (`user_id`) ON DELETE SET NULL,
  ADD CONSTRAINT `dms_documents_ibfk_doc_type` FOREIGN KEY (`doc_type`) REFERENCES `document_types` (`type_id`),
  ADD CONSTRAINT `dms_documents_ibfk_folder` FOREIGN KEY (`folder_id`) REFERENCES `folders` (`folder_id`) ON DELETE SET NULL,
  ADD CONSTRAINT `dms_documents_ibfk_received_by` FOREIGN KEY (`received_by_user_id`) REFERENCES `dms_user` (`user_id`) ON DELETE SET NULL,
  ADD CONSTRAINT `dms_documents_ibfk_reply` FOREIGN KEY (`is_reply_to_doc_id`) REFERENCES `dms_documents` (`doc_id`) ON DELETE SET NULL;

--
-- Constraints for table `dms_user`
--
ALTER TABLE `dms_user`
  ADD CONSTRAINT `dms_user_ibfk_department` FOREIGN KEY (`department_id`) REFERENCES `departments` (`department_id`) ON DELETE SET NULL;

--
-- Constraints for table `document_actions`
--
ALTER TABLE `document_actions`
  ADD CONSTRAINT `document_actions_ibfk_action` FOREIGN KEY (`action_id`) REFERENCES `action_required` (`action_id`) ON DELETE CASCADE,
  ADD CONSTRAINT `document_actions_ibfk_assigned_department` FOREIGN KEY (`assigned_to_department_id`) REFERENCES `departments` (`department_id`) ON DELETE SET NULL,
  ADD CONSTRAINT `document_actions_ibfk_assigned_user` FOREIGN KEY (`assigned_to_user_id`) REFERENCES `dms_user` (`user_id`) ON DELETE SET NULL,
  ADD CONSTRAINT `document_actions_ibfk_completed_by` FOREIGN KEY (`completed_by_user_id`) REFERENCES `dms_user` (`user_id`) ON DELETE SET NULL,
  ADD CONSTRAINT `document_actions_ibfk_created_by` FOREIGN KEY (`created_by_user_id`) REFERENCES `dms_user` (`user_id`) ON DELETE SET NULL,
  ADD CONSTRAINT `document_actions_ibfk_doc` FOREIGN KEY (`doc_id`) REFERENCES `dms_documents` (`doc_id`) ON DELETE CASCADE;

--
-- Constraints for table `document_departments`
--
ALTER TABLE `document_departments`
  ADD CONSTRAINT `document_departments_ibfk_department` FOREIGN KEY (`department_id`) REFERENCES `departments` (`department_id`) ON DELETE CASCADE,
  ADD CONSTRAINT `document_departments_ibfk_doc` FOREIGN KEY (`doc_id`) REFERENCES `dms_documents` (`doc_id`) ON DELETE CASCADE;

--
-- Constraints for table `document_roles`
--
ALTER TABLE `document_roles`
  ADD CONSTRAINT `document_roles_ibfk_doc` FOREIGN KEY (`doc_id`) REFERENCES `dms_documents` (`doc_id`) ON DELETE CASCADE;

--
-- Constraints for table `document_users`
--
ALTER TABLE `document_users`
  ADD CONSTRAINT `document_users_ibfk_doc` FOREIGN KEY (`doc_id`) REFERENCES `dms_documents` (`doc_id`) ON DELETE CASCADE,
  ADD CONSTRAINT `document_users_ibfk_user` FOREIGN KEY (`user_id`) REFERENCES `dms_user` (`user_id`) ON DELETE CASCADE;

--
-- Constraints for table `notifications`
--
ALTER TABLE `notifications`
  ADD CONSTRAINT `notifications_ibfk_doc` FOREIGN KEY (`related_doc_id`) REFERENCES `dms_documents` (`doc_id`) ON DELETE CASCADE;

--
-- Constraints for table `notification_departments`
--
ALTER TABLE `notification_departments`
  ADD CONSTRAINT `notification_departments_ibfk_department` FOREIGN KEY (`department_id`) REFERENCES `departments` (`department_id`) ON DELETE CASCADE,
  ADD CONSTRAINT `notification_departments_ibfk_notification` FOREIGN KEY (`notification_id`) REFERENCES `notifications` (`notification_id`) ON DELETE CASCADE;

--
-- Constraints for table `notification_reads`
--
ALTER TABLE `notification_reads`
  ADD CONSTRAINT `nr_fk_notification` FOREIGN KEY (`notification_id`) REFERENCES `notifications` (`notification_id`) ON DELETE CASCADE,
  ADD CONSTRAINT `nr_fk_user` FOREIGN KEY (`user_id`) REFERENCES `dms_user` (`user_id`) ON DELETE CASCADE;

--
-- Constraints for table `notification_roles`
--
ALTER TABLE `notification_roles`
  ADD CONSTRAINT `notification_roles_ibfk_notification` FOREIGN KEY (`notification_id`) REFERENCES `notifications` (`notification_id`) ON DELETE CASCADE;

--
-- Constraints for table `notification_users`
--
ALTER TABLE `notification_users`
  ADD CONSTRAINT `notification_users_ibfk_notification` FOREIGN KEY (`notification_id`) REFERENCES `notifications` (`notification_id`) ON DELETE CASCADE,
  ADD CONSTRAINT `notification_users_ibfk_user` FOREIGN KEY (`user_id`) REFERENCES `dms_user` (`user_id`) ON DELETE CASCADE;

--
-- Constraints for table `others`
--
ALTER TABLE `others`
  ADD CONSTRAINT `fk_others_created_by` FOREIGN KEY (`created_by_user_id`) REFERENCES `dms_user` (`user_id`) ON DELETE SET NULL ON UPDATE CASCADE,
  ADD CONSTRAINT `fk_others_updated_by` FOREIGN KEY (`updated_by_user_id`) REFERENCES `dms_user` (`user_id`) ON DELETE SET NULL ON UPDATE CASCADE;

--
-- Constraints for table `user_document_preferences`
--
ALTER TABLE `user_document_preferences`
  ADD CONSTRAINT `user_document_preferences_ibfk_doc` FOREIGN KEY (`doc_id`) REFERENCES `dms_documents` (`doc_id`) ON DELETE CASCADE,
  ADD CONSTRAINT `user_document_preferences_ibfk_user` FOREIGN KEY (`user_id`) REFERENCES `dms_user` (`user_id`) ON DELETE CASCADE;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
